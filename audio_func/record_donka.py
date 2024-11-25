import pyaudio
import numpy as np
from typing import List,Callable,Any,Tuple
import librosa.feature
import math
import wave
import time
import threading
import os
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt

DON = 0
KA  = 1

LEFT  = 0
RIGHT = 2

# Get noise statistics
def get_noise_statistics(noise_wav: np.ndarray) -> Tuple[float,float]:
    rms = librosa.feature.rms(y=noise_wav, frame_length=512, hop_length=512)[0,]
    noisemedian = np.percentile(rms, 50)
    sigma = np.percentile(rms, 84.1) - noisemedian

    return (noisemedian, sigma)

def play_donka_audio(donka_code: int, p: pyaudio.PyAudio|None=None):
    # Non-blocking audio playing
    def _thread():
        file = "audio/metronome/Don.wav" if donka_code%2==DON else "audio/metronome/Katsu.wav"
        audio = p if p!=None else pyaudio.PyAudio()
        with wave.open(file, 'rb') as wf:
            
            try:
                stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                                    channels=wf.getnchannels(),
                                    rate=wf.getframerate(),
                                    output=True)
                chunk = 512
                while len(data := wf.readframes(chunk)):  # Requires Python 3.8+ for :=
                    stream.write(data)

                stream.close()
            except OSError:
                pass
        if p==None:
            audio.terminate()
            
    thread = threading.Thread(target=_thread)
    thread.start()

def retrieve_audio_input(audio: np.ndarray, noise_stat: Tuple[float,float], sample_rate: float,
                         frame_left: int=1600, frame_right: int=3200) -> np.ndarray|None:
    # Use the first onset which is above the noise threshold
    onsets = librosa.onset.onset_detect(y=audio, sr=sample_rate, units="samples")
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0,]
    noise_med,noise_sig = noise_stat

    # Find onset with the maximum energy
    for onset in onsets:
        if onset >= frame_left and onset < len(audio) - frame_right \
            and rms[onset//512] >= noise_med + 3*noise_sig:
            return audio[onset-frame_left:onset+frame_right]
        


def record_inputs(donka_code: int,
                  in_device_idx: int|None=None, 
                  out_device_idx: int|None=None, 
                  sample_rate: int=16000, chunk_sz: int=512, 
                  p: pyaudio.PyAudio|None=None, 
                  count: int=8, noise_stat: Tuple[float,float]|None=None,
                  on_note_callback: Callable[[np.ndarray], Any]=lambda x: 0) -> List[np.ndarray]:
    # Load audio for metronome
    donka_type = donka_code % 2

    audio = p if p!=None else pyaudio.PyAudio()

    if in_device_idx==None:
        in_device_idx = audio.get_default_input_device_info()['index']
    if out_device_idx==None:
        out_device_idx = audio.get_default_output_device_info()['index']

    audio_arr: np.ndarray = np.zeros((sample_rate//2, ))

    in_stream = audio.open(format=pyaudio.paFloat32, channels=1,
                           rate=sample_rate, input=True,
                           input_device_index=in_device_idx,
                           frames_per_buffer=chunk_sz)
    
    # Use 1 second as a representative of noise
    if noise_stat==None:
        noise_arr: np.ndarray = np.zeros((sample_rate))
        for i in range(0, sample_rate, chunk_sz):
            chunk = np.frombuffer(in_stream.read(chunk_sz), dtype=np.float32)
            noise_arr[i:i+chunk_sz] = chunk[:min(chunk_sz, sample_rate - i)]

        noise_stat = get_noise_statistics(noise_arr)

    # Record audio inputs
    input_samples: List[np.ndarray] = []

    timing = 0

    while len(input_samples) < count:
        if len(audio_arr)//2 <= timing and timing < len(audio_arr)//2 + chunk_sz:
            play_donka_audio(donka_type, p=audio)

        # Read next chunk
        chunk = np.frombuffer(in_stream.read(chunk_sz), dtype=np.float32)

        # Shift audio array
        audio_arr[:-chunk_sz] = audio_arr[chunk_sz:]
        audio_arr[-chunk_sz:] = chunk

        # Retrieve input once a new audio has been fully consumed
        rms = librosa.feature.rms(y=audio_arr[:2048], frame_length=512, hop_length=512)[0,]
        if rms[4] >= noise_stat[0] + 3*noise_stat[1]:
            note = retrieve_audio_input(audio_arr, noise_stat, sample_rate)
            if type(note)==np.ndarray and len(note) > 0:
                input_samples.append(note.copy())
                on_note_callback(note.copy())
                audio_arr[:len(note) + 2048] = 0
                #audio_arr = np.zeros_like(audio_arr)

        timing += chunk_sz
        timing %= len(audio_arr)

    # Cleanup
    in_stream.stop_stream()
    in_stream.close()
    if p==None:
        audio.terminate()

    return input_samples


def main(in_device_idx: int|None=None, 
         sample_rate: int=16000,
         chunk_sz: int=512, 
         channels: int=1,
         verbose=False, target_dir="audio/user"):
    def log(message, tag: str|None=None):
        if verbose:
            if not tag:
                print(f"[LOG] record_donka.main(): {message}")
            if tag:
                print(f"[{tag}] record_donka.main(): {message}")

    p = pyaudio.PyAudio()
    
    # Get devices
    if in_device_idx==None:
        in_device_idx = p.get_default_input_device_info()['index']
    out_device_idx = p.get_default_output_device_info()['index']
    log("Using input device " + p.get_device_info_by_host_api_device_index(0, in_device_idx).get('name'))
    log("Using output device " + p.get_device_info_by_host_api_device_index(0, out_device_idx).get('name'))

    # Use 1 second as a representative of noise
    log("Be quiet!", tag="USER-INPUT")
    in_stream = p.open(format=pyaudio.paFloat32, channels=1,
                        rate=sample_rate, input=True,
                        input_device_index=in_device_idx,
                        frames_per_buffer=chunk_sz)
    
    noise_arr: np.ndarray = np.zeros((sample_rate))
    for i in range(0, sample_rate, chunk_sz):
        chunk = np.frombuffer(in_stream.read(chunk_sz), dtype=np.float32)
        noise_arr[i:i+chunk_sz] = chunk[:min(chunk_sz, sample_rate - i)]

    noise_stat = get_noise_statistics(noise_arr)

    in_stream.stop_stream()
    in_stream.close()

    wf.write(os.path.join(target_dir, f"noise.wav"), sample_rate, noise_arr)

    # Map audio inputs to right-left dons/kas
    params = [(volume, side, donka) 
              for donka in [DON, KA]
              for side in [RIGHT, LEFT]
              for volume in ["quiet", "medium", "loud"]
             ]
    for volume, side, donka in params:
        donka_string = "Don" if donka==DON else "Ka"
        side_string = "left" if side==LEFT else "right"

        log(f"Play {side_string} {donka_string}s with {volume} volume to the beat!", tag="USER-INPUT")
        notes = record_inputs(donka|side, in_device_idx=in_device_idx, 
                              out_device_idx=out_device_idx, sample_rate=sample_rate,
                              chunk_sz=chunk_sz, p=p, noise_stat=noise_stat,
                              on_note_callback=lambda x: log(" - Detected!"))
        for idx,note in enumerate(notes):
            wf.write(os.path.join(target_dir, f"{volume}_{donka|side}_{idx}.wav"), sample_rate, note)
        time.sleep(0.5)

    p.terminate()



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="DonKa Recorder",
        description="Retrieve audio inputs from users by following a metronome."
    )

    parser.add_argument("--in-device-idx", type=int, default=-1)

    args = parser.parse_args()
    in_device_idx = args.in_device_idx

    if in_device_idx < 0:
        in_device_idx = None

    main(in_device_idx=in_device_idx, verbose=True)