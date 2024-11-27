from typing import List,Tuple,Callable,Any
import numpy as np
import librosa
import random
from sklearn.neighbors import KNeighborsClassifier
import pyaudio
import os
import os.path
import struct
import time
import matplotlib.pyplot as plt

from utility import compute_mel_rep,DonkaCode,AudioStatistics,retrieve_audio_inputs,\
    DON,KA,LEFT,RIGHT
from validate_input import KNNHyperParamMetric

import threading

def note_detect(audio: np.ndarray, sr: int, train_mel: List[np.ndarray], train_y: np.ndarray, frame_left: int=1600, frame_right: int=3200, K=5, normalize=True) -> List[Tuple[int, int]]:
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, units="samples")

    random.seed(123)

    # Calculate RMS energy per frame.
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=512)[0,]
    envtm = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    # Use first second as a representative of background noise
    noiseidx = envtm <= 1
    noisemedian = np.percentile(rms[noiseidx], 50)
    sigma = np.percentile(rms[noiseidx], 84.1) - noisemedian

    threshold = noisemedian + 5*sigma
    
    # Transform training data
    if normalize: train_mel: List[np.ndarray] = [i/np.sum(i) for i in train_mel]

    last_onset = -frame_left

    answer = []

    knn = KNeighborsClassifier(K)
    knn.fit(train_mel, [np.argmax(i) for i in train_y])

    # Classify each onset
    for onset in onsets:
        # Likely detecting the previous onset
        if onset < last_onset + frame_left: continue

        note = audio[onset-frame_left:onset+frame_right]
        # Invalid position
        if note.shape != (frame_left + frame_right,):
            continue
        # Likely background noise
        if rms[onset//512] <= threshold:
            continue

        note_mel = compute_mel_rep(note)
        if normalize: note_mel /= np.sum(note_mel)

        # Problem-specific vote system
        votes = knn.predict_proba([note_mel.flatten()])[0]
        if votes[0] + votes[1] > votes[2] + votes[3]:
            answer.append((onset, np.argmax(votes[:2])))
        elif votes[0] + votes[1] < votes[2] + votes[3]:
            answer.append((onset, np.argmax(votes[2:4]) + 2))
        else:
            answer.append((onset, np.argmax(votes)))
        last_onset = onset
    return answer

class AudioInputDetector:
    def __init__(self, 
                 train_x: List[np.ndarray], 
                 train_y: np.ndarray, 
                 noise_stat: AudioStatistics,  # A representative of the noise
                 detect_callback: Callable[[DonkaCode, float],Any],  # detect_callback(which_detected, time_detected)
                 frame_left: int=1600, frame_right: int=3200,
                 in_device_idx: int = -1,
                 sample_rate: int=16000, chunk_sz=512, buffer_sz: int=16000,
                 audio_interface: pyaudio.PyAudio|None=None,
                 val_data_dir: str=".val_cache",
                 verbose: bool=False
                ):
        self.verbose = verbose
        self.running = False

        # Hyperparameters
        self.frame_left,self.frame_right = frame_left,frame_right

        # Load validation data
        val_results: KNNHyperParamMetric = {}
        try:
            with open(os.path.join(val_data_dir, "knn"), "rb") as f:
                while chunk := f.read(struct.calcsize("I?ddd")):
                    k,is_norm,note_rate,bin_acc,base_acc = struct.unpack("I?ddd", chunk)
                    if (k, is_norm) not in val_results:
                        val_results[(k, is_norm)] = {}
                    val_results[(k, is_norm)][note_rate] = (bin_acc, base_acc)

        except FileNotFoundError:
            self._log(f"KNN Validation results not found!", 
                      tag="ERROR", method="__init__")
            raise FileNotFoundError("Validation results not found. Make sure to run \"python audio_func/validate_input.py\" before running detection.")
        
        # ML Preprocessing
        self.train_mel: np.ndarray = np.array([compute_mel_rep(x, sr=sample_rate) for x in train_x])
        self.train_y = train_y

        self.noise_stat: AudioStatistics = noise_stat
        self.detect_callback = detect_callback
        self._log(f"Loaded ML processing data.",
                  tag="INFO", method="__init__")

        # Audio streaming
        self.sample_rate = sample_rate
        self.buffer_sz,self.chunk_sz = buffer_sz,chunk_sz
        self.audio_buffer = np.zeros((buffer_sz,))
        self.time_recv = np.zeros((buffer_sz // self.chunk_sz, ))  # Time the chunk was received. This assumes that the last sample in the chunk was recorded at that time

        self.audio_interface: pyaudio.PyAudio = audio_interface if audio_interface else pyaudio.PyAudio()
        self.in_device_idx = self.audio_interface.get_default_input_device_info()['index'] if in_device_idx < 0 else in_device_idx

        # Real-time operations. These are updated on every read from the audio stream
        self.rms_frame_length, self.rms_hop_length=self.chunk_sz,self.chunk_sz
        self.rms = librosa.feature.rms(y=self.audio_buffer, frame_length=self.rms_frame_length, hop_length=self.rms_hop_length)[0,]

        self.last_10_note_sample: np.ndarray = np.zeros((10, )) - 1e9
        self.last_10_note_min_energy: np.ndarray = np.zeros((10, )) - 1

    def detect_note(self) -> List[Tuple[DonkaCode,float]]:
        onsets = librosa.onset.onset_detect(y=self.audio_buffer, sr=self.sample_rate, units="samples")

        result = []
        for onset in onsets:
            # Not enough data
            if onset < self.frame_left or onset >= self.buffer_sz - self.frame_right:
                continue
            # Noise
            if self.rms[onset//self.rms_hop_length] <= self.noise_stat.get_energy_median() + 3*self.noise_stat.get_energy_sigma():
                continue
            
            if onset <= self.last_10_note_sample[-1] + self.frame_left:
                continue

            note = self.audio_buffer[onset-self.frame_left:onset+self.frame_right].copy()
            
            # Classify onset
            cls2donka_code = [
                DON|RIGHT,
                DON|LEFT,
                KA|RIGHT,
                KA|LEFT,
            ]
            cls_distance: List[Tuple[float, int]] = []
            s = time.time()
            note_mel = compute_mel_rep(note, self.sample_rate)
            for other_note_mel,cls in zip(self.train_mel, self.train_y):
                cls = np.argmax(cls)
                cls_distance.append((np.sum((other_note_mel - note_mel)**2)**0.5, cls))

            cls_distance.sort()
            donka_code = cls2donka_code[cls_distance[0][1]]
            
            note_time = self.time_recv[onset // self.chunk_sz]

            note_time -= self.chunk_sz / self.sample_rate
            note_time += (onset % self.chunk_sz) / self.sample_rate

            result.append((donka_code, note_time, onset))
        
        return result
        
    def add_chunk(self, chunk: np.ndarray, chunk_time: float):
        assert chunk.shape == (self.chunk_sz,)

        # Shift buffers
        self.time_recv[:-1] = self.time_recv[1:]
        self.audio_buffer[:-self.chunk_sz] = self.audio_buffer[self.chunk_sz:].copy()
        
        # Add chunk
        self.time_recv[-1] = chunk_time
        self.audio_buffer[-self.chunk_sz:] = chunk.copy()
        self.rms = librosa.feature.rms(y=self.audio_buffer, frame_length=self.rms_frame_length, hop_length=self.rms_hop_length)[0,]
        self.last_10_note_min_energy[-1] = min(self.last_10_note_min_energy[-1], self.rms[-1])

        # Timing
        self.last_10_note_sample -= self.chunk_sz

    def run(self):
        in_stream = self.audio_interface.open(
            format=pyaudio.paFloat32, channels=1,
            rate=self.sample_rate, input=True,
            input_device_index=self.in_device_idx,
            frames_per_buffer=self.chunk_sz)
        self._log(f"Using input device {self.audio_interface.get_device_info_by_host_api_device_index(0, self.in_device_idx).get('name')}.",
                  tag="INFO", method="__init__")
        
        while self.running:
            chunk = np.frombuffer(in_stream.read(self.chunk_sz), dtype=np.float32)
            self.add_chunk(chunk, time.time())

            for donka_code, note_time, onset in self.detect_note():
                self.last_10_note_sample[:-1] = self.last_10_note_sample[1:].copy()
                self.last_10_note_sample[-1] = onset
                
                self.detect_callback(donka_code, note_time)


        in_stream.stop_stream()
        in_stream.close()

    def start(self) -> threading.Thread:
        self.running = True
        thread = threading.Thread(target=lambda: self.run())

        thread.start()
        return thread

    def stop(self):
        self.running = False

    def _log(self, message: Any, tag: str="LOG", method: str="unknown"):
        if self.verbose:
            print(f"[{tag}] {self}.{method}: {message}")

def main():
    train_x,train_y,noise_stat = retrieve_audio_inputs()
    detector = AudioInputDetector(train_x, train_y, noise_stat, print)

    thread = detector.start()
    input()

    detector.stop()

    thread.join()

if __name__=="__main__":
    main()