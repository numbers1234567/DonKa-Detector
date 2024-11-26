import librosa
from typing import Tuple, List, Dict
import numpy as np
import os
from functools import cache


type DonkaCode = int
DON: DonkaCode = 0
KA: DonkaCode  = 1

LEFT: DonkaCode  = 0
RIGHT: DonkaCode = 2

class AudioStatistics:
    # Compute statistics about audio upfront
    def __init__(self, audio_wav: np.ndarray):
        rms = librosa.feature.rms(y=audio_wav, frame_length=512, hop_length=512)[0,]
        self.energy_median = np.percentile(rms, 50)
        self.energy_sigma = np.percentile(rms, 84.1) - self.energy_median
        self.wav_std = np.std(audio_wav)

    def get_energy_median(self): return self.energy_median.copy()
    def get_energy_sigma(self): return self.energy_sigma.copy()
    def get_wav_std(self): return self.wav_std.copy()


# Get noise statistics
def get_noise_statistics(noise_wav: np.ndarray) -> Tuple[float,float]:
    rms = librosa.feature.rms(y=noise_wav, frame_length=512, hop_length=512)[0,]
    noisemedian = np.percentile(rms, 50)
    sigma = np.percentile(rms, 84.1) - noisemedian

    return (noisemedian, sigma)

def compute_mel_rep(note: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.melspectrogram(y=note, sr=sr, n_mels=128, fmax=8000)

def retrieve_audio_inputs(target_dir: str="audio/user", sr: int=16000) -> Tuple[List[np.ndarray], np.ndarray, AudioStatistics]:
    donka_code2class = {
        DON|RIGHT : 0,
        DON|LEFT : 1,
        KA| RIGHT : 2,
        KA| LEFT : 3,
    }
    note_x_param: Dict[Tuple[str, int], List[np.ndarray]] = {}
    note_x: List[np.ndarray] = []
    note_y: np.ndarray = np.zeros((96,4))

    # Estimate of noise
    noise_arr, sr = librosa.load(os.path.join(target_dir, "noise.wav"), sr=sr)

    # Retrieve proper inputs
    audio_input_files = os.listdir(target_dir)
    for file in audio_input_files:
        if file=="noise.wav":
            continue
        try:
            audio_arr, sr = librosa.load(os.path.join(target_dir, file), sr=sr)
            
            [volume,donka_code,idx] = file[:-4].split("_")
            donka_code,idx = int(donka_code),int(idx)
            if (volume,donka_code) not in note_x_param:
                note_x_param[(volume,donka_code)] = []
            note_x_param[(volume,donka_code)].append(audio_arr)

        except ValueError as e:
            print(f"Failed to load input file {file}.")
            print(e)

    # Set training/testing data
    idx = 0
    for (volume,donka_code),notes in note_x_param.items():
        note_x += notes
        note_y[idx:idx+len(notes),donka_code2class[donka_code]] = 1
        idx += len(notes)

    return note_x,note_y,AudioStatistics(noise_arr)