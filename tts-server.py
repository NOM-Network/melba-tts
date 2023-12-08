
import torch
import wave
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import nltk
nltk.download('punkt')
import random
random.seed(0)

import numpy as np
np.random.seed(0)


# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)

import dotenv
configs = dotenv.dotenv_values("./.env")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
config = yaml.safe_load(open(configs['CONFIG']))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]
params_whole = torch.load(configs["MODEL"], map_location='cpu')
params = params_whole['net']
for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

def inference(text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, speed=1.0,return_device='cpu'):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, 
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        print(f"speed: {speed}")
        duration = model.predictor.duration_proj(x) / speed

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    
    start = time.perf_counter() 
    if return_device == 'cpu':   
        out = out.squeeze().cpu().numpy()[..., :-90] # weird pulse at the end of the model, need to be fixed later
    
    # if return_device == 'cpu':
    #     out = out.cpu()
    elif return_device == 'cuda':
        # check if is actually on cuda
        if out.is_cuda:
            # remove last 90 samples while still on cuda
            out = out.squeeze()[..., :-90]


    end = time.perf_counter()
    print("Time taken to convert to numpy:", (end - start) * 1000, "ms")


    return out
import numpy as np
import scipy.stats as stats
import librosa



# async websocket server
import asyncio
import websockets
import json
import base64
import os
import time
from scipy.signal import resample
def pitch_shift(audio_array, factor):
    # Calculate the new length of the audio array after pitch shifting
    new_length = int(len(audio_array) / factor)

    # Use scipy's resample function to pitch up the audio
    pitched_up_array = resample(audio_array, new_length)

    return pitched_up_array


from scipy.interpolate import interp1d


import io

from fractions import Fraction
from scipy.io import wavfile
# wavfile.read("If_you_aren-t_subscribed_3.wav")
import soundfile as sf


from audiostretchy.stretch import AudioStretch


# at /synthesize
async def synthesize(websocket, path):
    data = await websocket.recv()
    print(data)
    json_data = json.loads(data)
    text = json_data['text']
    alpha = json_data['alpha'] if "alpha" in json_data else 0.3
    beta = json_data['beta'] if "beta" in json_data else 0.7
    speed = json_data['speed'] if "speed" in json_data else 1.0
    print("speed: ", speed)
    override_alpha = False
    override_beta = False
    try:
        # replace ellipsis with three dots
        text = text.replace('…', '...')
        # replace three dots with a period
        text = text.replace('...', '.')
        # replace hyphens with spaces
        text = text.replace('-', ' ')
        # replace underscores with spaces
        text = text.replace('_', ' ')
        # replace double quotes with single quotes
        text = text.replace('"', "'")
        # replace double spaces with single spaces
        text = text.replace('  ', ' ')
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = text.replace("\t", " ")
        text = text.replace("  ", " ")
        text = text.replace("*", "")

        

        text = text.strip()

        # check if text is less than 10 characters and does not end with a punctuation
        if len(text) < 20 and text[-1] not in ['.', '!', '?']:
            text += '.'
            override_alpha = True
            override_beta = True

        # add a punctuation if text does not end with a punctuation
        if text[-1] not in ['.', '!', '?', ',', ':', ';']:
            text += '.'

        # filter out control characters like \n \r \t, etc.
        text = ''.join([i if ord(i) < 128 else ' ' for i in text])
        print("Actual text: ", text)

        # check that there is non repeating non alphanumeric characters
        consecutive = 0
        for i in range(len(text) - 1):
            if text[i] == text[i + 1] and not text[i].isalnum():
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= 6:
                override_alpha = True
                override_beta = True
                break

        if len(text) < 10:
            override_alpha = True
            override_beta = True
    except:
        # raise server error if text is not a string
        override_alpha = True
        override_beta = True

    # clamp to 0.0 to 1.0
    alpha = max(0.0, min(1.0, alpha)) if not override_alpha else 0.0
    beta = max(0.0, min(1.0, beta)) if not override_beta else 0.0

    print("alpha: ", alpha)
    print("beta: ", beta)


    ref_file = json_data['ref_file'] if "ref_file" in json_data else "default.wav"
    try:
        if os.path.exists("./Samples/" + ref_file):
            ref_s = compute_style("./Samples/" + ref_file)
        else:
            # download file
            import requests
            r = requests.get(ref_file, allow_redirects=True)
            fname = ref_file.split('/')[-1]
            open("./Samples/" + fname, 'wb').write(r.content)
            ref_s = compute_style("./Samples/" + fname)
    except:
        ref_s = compute_style("default.wav")


    start = time.perf_counter()
 
    wav = inference(text, ref_s, diffusion_steps=10, alpha=alpha, beta=beta, embedding_scale=1.0, speed=1.0, return_device='cpu')
    end = time.perf_counter()
    print("Inference time: ", (end - start) * 1000, "ms")
    # convert to floast32 to int16
    wav = wav.astype(np.float32)

    # get pitch
    pitch = json_data['pitch'] if "pitch" in json_data else 1.0
    pitch = float(pitch)
    print(pitch)
    if pitch != 0.0:
        wav = pitch_shift(wav, pitch)
        # wav = pitchshifter.shiftpitch(wav, pitch)

    # get speed
    speed = json_data['speed'] if "speed" in json_data else 1.0
    speed = float(speed)
    print(speed)
    if speed != 1.0:
        audio_stretch = AudioStretch()
        byte_wav = io.BytesIO()
        # write to byte_wav
        sf.write(byte_wav, wav, 24000, format='wav')
        byte_wav.seek(0)

        # stretch
        audio_stretch.open(file=byte_wav, format='wav')
        audio_stretch.stretch(ratio=speed)
        output = io.BytesIO()
        audio_stretch.save_wav(output,close=False)
        output.seek(0)
        wav, sr = sf.read(output)        


    # upsample to 40000
    wav = librosa.resample(wav, orig_sr=24000, target_sr=40000)

    wav =(wav * 32768.0).astype(np.int16)


    wav = base64.b64encode(wav.tobytes())
    buffer_size = 8000
    for i in range(0, len(wav), buffer_size):
        await websocket.send(wav[i:i+buffer_size])


               
print("Server is starting...")

# print fast stretch and pitch shifts with sample rate 24000

# start websocket server
start_server = websockets.serve(synthesize, "0.0.0.0", 8767)
asyncio.get_event_loop().run_until_complete(start_server)

if __name__ == '__main__':
    asyncio.get_event_loop().run_forever()