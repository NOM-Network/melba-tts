from flask import Flask, render_template, request, send_file
import os


# Create a queue to hold incoming requests
import threading
lock = threading.Lock()
import time
import os
import numpy as np
import sys
from tempfile import NamedTemporaryFile
import asyncio
import soundfile
import resampy
# import pyaudio
import websockets
import base64
import json

pitch = 0.0
speed = 1.0
alpha = 0.3
beta = 0.7
ref_file = "default.wav"

async def style_tts(text,timings=False):
    # same as hello
    # uri = "ws://71.158.89.73:4436"
    
    uri = "ws://localhost:8767"
    async with websockets.connect(uri) as websocket:
        # send the audio file
        json_test = json.dumps({'text': text, "pitch": pitch/10, "speed": speed, "alpha": alpha, "beta": beta, "ref_file": ref_file})
        await websocket.send(json_test)
        # receive the audio file
        wav_sum = b''
        while True:
            try:
                wav = await websocket.recv()
                if wav == b'':
                    break
                wav_sum += wav
            except websockets.exceptions.ConnectionClosed:
                break
        wav = base64.b64decode(wav_sum)
        numpy_wav = np.frombuffer(wav, dtype=np.int16)
        # convert to int16
        # numpy_wav = (numpy_wav * 32767).astype(np.int16)
        print(numpy_wav.shape)
        # write the audio file
    return numpy_wav.tobytes()

def tts(text,stack, timings=False, subtitle_window=None):
    # with lock:
    # split into sentences using . ? , and !
    sentences = []
    current_sentence = ""
    overflow = ""
    for char in text:
        current_sentence += char
        if char in [".", "?", "!"]:
            sentences.append(current_sentence)
            current_sentence = ""
            overflow = ""
        # account for the case where there is no punctuation at the end of the text
        else:
            overflow = current_sentence
    # add the overflow to the last sentence
    if len(overflow) > 0:
        sentences.append(overflow)
    # check if there are no punctuation in the input text
    if len(sentences) == 0:
        sentences.append(text)

    if stack not in ["style-tts"]:
        raise ValueError(f"Invalid tts backend: {stack}")
        
    # remove any single characters
    sentences = [sentence for sentence in sentences if len(sentence) > 1]

    # if the original text didnt have ant punctuation, just tts the whole thing
    if len(sentences) == 0:
        sentences = [text]

    

    stack, other = stack.split("+") if "+" in stack else (stack, None)

    audio_bytes_sum = b""
    # tts each sentence
    start_tts = time.perf_counter()
    first = True
    for sentence in sentences:
        audio_bytes = b""
        if stack == "style-tts":
            audio_bytes = asyncio.run(style_tts(sentence,timings=timings))
        else:
            raise ValueError(f"Invalid tts backend: {stack}")
        
      
        first_tts_time = time.perf_counter()
        if first:
            print(f"first tts chunk: {(first_tts_time-start_tts)*1000:.2f} ms")
            first = False
        audio_bytes_sum += audio_bytes
    if timings:
        print(f"tts: {(time.perf_counter()-start_tts)*1000:.2f} ms")


    # write to a output wav
    audio_np = np.frombuffer(audio_bytes_sum, dtype=np.int16)


    return audio_np
def chunk_sentence(sentence, n):
    if len(sentence) <= n:
        return [sentence]
    words = sentence.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > n:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    chunks.append(" ".join(current_chunk))
    print(len(' '.join(chunks)))
    print(len(sentence))
    # assert len(' '.join(chunks)) == len(sentence), "Chunks do not add up to original sentence"

    return chunks
def concat_short_sentences(sentences, min_length):
    result = []
    current_sentence = sentences[0]

    for next_sentence in sentences[1:]:
        if len(current_sentence) < min_length:
            current_sentence += ' ' + next_sentence
        else:
            result.append(current_sentence)
            current_sentence = next_sentence

    result.append(current_sentence)  # Add the last sentence

    final_result = []
    current_concatenation = result[0]

    for sentence in result[1:]:
        if len(current_concatenation) + len(sentence) + 1 < min_length:
            current_concatenation += ' ' + sentence
        else:
            final_result.append(current_concatenation)
            current_concatenation = sentence

    final_result.append(current_concatenation)  # Add the last concatenated sentence

    return final_result

app = Flask(__name__)
import io
import uuid
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/synthesize', methods=['POST'])
def synthesize():
    global pitch, speed, alpha, beta, ref_file

    start_full = time.perf_counter()
    # print the form data to the console
    print(f"request.form: {request.form}")

    text = request.form['text']
    selected_voice = request.form['voice']
    pitch = request.form['pitch']
    pitch2 = request.form['pitch2']
    speed = request.form['speed']

    # temperature = request.form['temperature']
    # length_penalty = request.form['length_penalty']
    alpha = request.form['alpha']
    beta = request.form['beta']
    print(f"beta: {beta}")
    print(f"alpha: {alpha}")
    print(f"pitch: {pitch}")
    print(f"speed: {speed}")
    print(f"voice: {selected_voice}")
    backend = "piper"

    # get the reference file name from the select dropdown
    ref_id = request.form['ref_clip']
    print(f"ref_id: {ref_id}")
    # map to ref_file_name
    ref_file = ref_id + ".wav"
    ref_file = "default.wav"
    # cap pitch between -12.0 and 24.0
    pitch = min(24.0, max(-12.0, float(pitch)))
    # cap speed between 1.0 and 3.0
    speed = min(3.0, max(0.1, float(speed)))

    pitch2 = min(24.0, max(-12.0, float(pitch2)))

    alpha = min(1.0, max(0.0, float(alpha)))
    beta = min(1.0, max(0.0, float(beta)))




    if selected_voice == "voice2":
        backend = "style-tts"
        sample_rate = 40000

    # sanitize text, limit to alphanumeric and punctuation
    # text = ''.join([c for c in text if c.isalnum() or c in [' ', '.', '?', '!', ',','\'']])
    # For example:
    print(f"tts: {text}")
    print()
    tts_outputs = []
    split_tezt = chunk_sentence(text, 300)
    split_tezt = concat_short_sentences(split_tezt, 20)
    print(f"split text: {split_tezt}", len(split_tezt))
    print()
    for sentence in split_tezt:
        print(f"sentence: {sentence}")
        print()
        tts_output = tts(sentence,stack=backend)
        if tts_output is None:
            # send the error message with code 500
            return "Error: TTS failed", 500


        tts_outputs.append(tts_output)

    tts_output_sum = np.concatenate(tts_outputs)
    
    # Write the tts_output to a temporary wav file
    # Save the output to a temporary WAV file
    # tts_output_sum = tts(text,stack=backend)
    # in memory file then to binaryIO for send_file
    memory_file = io.BytesIO()
    
    soundfile.write(memory_file, tts_output_sum, sample_rate, format="WAV")
    memory_file.seek(0)
    download_name = uuid.uuid4().hex
    # Send the temporary WAV file to the user
    print("sending file")
    print(f"full: {(time.perf_counter()-start_full)*1000:.2f} ms")
    return send_file(memory_file, as_attachment=True, download_name=f"{download_name}.wav", mimetype="audio/wav")

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5326,debug=False)
