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
codec = None
bitrate = 64

async def style_tts(text,timings=False):
    # same as hello

    wav_sum = b''
    uri = "ws://localhost:8767"
    async with websockets.connect(uri) as websocket:
        # send the audio file
        json_test = json.dumps({'text': text, "pitch": pitch/10, "speed": speed, "alpha": alpha, "beta": beta, "ref_file": ref_file, "codec": codec, "bitrate": bitrate})
        await websocket.send(json_test)
        # receive the audio file
        
        while True:
            try:
                wav = await websocket.recv()
                if wav == b'':
                    break
                wav_sum += wav
            except websockets.exceptions.ConnectionClosed:
                break

    return wav_sum
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

    audio_bytes_sum = []
    # tts each sentence
    start_tts = time.perf_counter()
    first = True
    for sentence in sentences:
        print(f"sentence: {sentence}")
        audio_bytes = b""
        if stack == "style-tts":
            audio_bytes = asyncio.run(style_tts(sentence,timings=timings))
        else:
            raise ValueError(f"Invalid tts backend: {stack}")
        
      
        first_tts_time = time.perf_counter()
        if first:
            print(f"first tts chunk: {(first_tts_time-start_tts)*1000:.2f} ms")
            first = False
        audio_bytes_sum.append(audio_bytes)
    if timings:
        print(f"tts: {(time.perf_counter()-start_tts)*1000:.2f} ms")

    # with open("tts.wav", "wb") as f:
        # f.write(audio_bytes_sum)
    # write to a output wav
    # audio_np = np.frombuffer(audio_bytes_sum, dtype=np.int16)
    assert type(audio_bytes_sum) == list, "audio_bytes_sum is not a list"
    assert type(audio_bytes_sum[0]) == bytes, "audio_bytes_sum[0] is not bytes"
    audio_np = concat_audio(audio_bytes_sum)

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

import ffmpeg
import wave
def concat_wav_files(wav_bytes_list):
    start = time.perf_counter()
    # Create a BytesIO object to hold the concatenated audio data
    output_bytes = io.BytesIO()
    print(f"wav_bytes_list: {len(wav_bytes_list)}")

    # Initialize variables for the parameters of the first WAV file
    num_channels = None
    sample_width = None
    frame_rate = None

    # Iterate through each WAV file in the list
    for wav_bytes in wav_bytes_list:
        # Create a BytesIO object from the current WAV file bytes
        wav_file = io.BytesIO(wav_bytes)

        # Create a Wave_read object from the BytesIO object
        with wave.open(wav_file, 'rb') as wave_read:
            # Read parameters from the first WAV file if not initialized yet
            if num_channels is None:
                num_channels = wave_read.getnchannels()
                sample_width = wave_read.getsampwidth()
                frame_rate = wave_read.getframerate()

            # Ensure that the parameters of all WAV files are the same
            assert wave_read.getnchannels() == num_channels
            assert wave_read.getsampwidth() == sample_width
            assert wave_read.getframerate() == frame_rate

            # Read audio data from the current WAV file and write to the output BytesIO object
            output_bytes.write(wave_read.readframes(wave_read.getnframes()))

    # Reset the position of the output BytesIO object to the beginning
    output_bytes.seek(0)
    print(f"output_bytes: {len(output_bytes.getvalue())}")
    print(f"num_channels: {num_channels}")
    print(f"sample_width: {sample_width}")  
    print(f"frame_rate: {frame_rate}")
    # Create a new Wave_write object for the concatenated WAV file
    with wave.open(output_bytes, 'wb') as wave_write:
        # Set the parameters of the concatenated WAV file
        wave_write.setnchannels(num_channels)
        wave_write.setsampwidth(sample_width)
        wave_write.setframerate(frame_rate)
        wave_write.setnframes(len(output_bytes.getvalue()) // (num_channels * sample_width))
    print(f"concat_wav_files: {(time.perf_counter()-start)*1000:.2f} ms")
    # Return the bytes of the concatenated WAV file
    return output_bytes.getvalue()

def concat_audio(tts_outputs: list):
    stat = time.perf_counter()
    # tt_outputs is a list of audio files as bytes
    if len(tts_outputs) == 1:
        return tts_outputs[0]
    print(len(tts_outputs))
    global codec

    # Create a list of input audio streams for ffmpeg
    ffmpeg_codec  = None
    ffmpeg_format = None
    input = None
    if codec == "wav":
        return concat_wav_files(tts_outputs) # ugly hack that kinda doesn't work but oh well
    elif codec == "opus":
        ffmpeg_codec = "libopus"
        ffmpeg_format = "ogg"
        input = ffmpeg.input('pipe:', loglevel='error', f=ffmpeg_format, acodec="libopus")
    elif codec == "vorbis":
        ffmpeg_codec = "libvorbis"
        ffmpeg_format = "ogg"
        input = ffmpeg.input('pipe:', loglevel='error', f=ffmpeg_format, acodec="libvorbis")
    elif codec == "flac":
        ffmpeg_codec = "flac"
        ffmpeg_format = "flac"
        input = ffmpeg.input('pipe:', loglevel='error', f=ffmpeg_format, acodec="flac")
    else:
        raise ValueError(f"Invalid codec: {codec}")
    print(type(tts_outputs[0]))


    # Run ffmpeg to concatenate the audio streams
    # concatenated_audio_bytes, _ = ffmpeg.output(*input_streams, "pipe:", map='a', f=ffmpeg_format, loglevel='error',).run(input=b''.join(tts_outputs), capture_stdout=True, capture_stderr=False)
    concat,_ = ffmpeg.concat(input, n=1, v=0, a=1).output('pipe:',format=ffmpeg_format, loglevel='error').run(input=b''.join(tts_outputs), capture_stdout=True, capture_stderr=False)
    print(f"concat: {len(concat)}")
    # with open(f"concat.{codec}", "wb") as f:
    #     f.write(concat)
    end = time.perf_counter()
    print(f"concat_audio: {(end-stat)*1000:.2f} ms")
    return concat

app = Flask(__name__)
import io
import uuid
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/synthesize', methods=['POST'])
def synthesize():
    global pitch, speed, alpha, beta, ref_file, codec, bitrate

    start_full = time.perf_counter()
    # print the form data to the console
    print(f"request.form: {request.form}")

    text = request.form['text']
    selected_voice = request.form['voice']
    pitch = request.form['pitch']
    pitch2 = request.form['pitch2']
    speed = request.form['speed']

    codec = request.form['codec'] or "wav"
    bitrate = request.form['bitrate'] or 64

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
   
    
    memory_file = concat_audio(tts_outputs)

    download_name = uuid.uuid4().hex
    print("outer chunking len")
    print(len(tts_outputs))
    # Send the temporary WAV file to the user
    print("sending file")
    print(f"full: {(time.perf_counter()-start_full)*1000:.2f} ms")
    file_extension = "wav"
    if codec == "mp3":
        file_extension = "mp3"
    elif codec == "opus" or codec == "vorbis":
        file_extension = "ogg"
    elif codec == "flac":
        file_extension = "flac"
    memory_file_from_bytes = io.BytesIO(memory_file)
    return send_file(memory_file_from_bytes, as_attachment=True, download_name=f"{download_name}.{file_extension}", mimetype=f"audio/{codec}")

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5326,debug=False)
