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

if __name__=="__main__":
    from utility import compute_mel_rep,DonkaCode,AudioStatistics,retrieve_audio_inputs,\
        DON,KA,LEFT,RIGHT
    from validate_input import KNNHyperParamMetric
else:
    from .utility import compute_mel_rep,DonkaCode,AudioStatistics,retrieve_audio_inputs,\
        DON,KA,LEFT,RIGHT
    from .validate_input import KNNHyperParamMetric

import threading

notes = []

class AudioInputDetector:
    def __init__(self, 
                 train_x: List[np.ndarray], 
                 train_y: np.ndarray, 
                 noise_stat: AudioStatistics,  # A representative of the noise
                 detect_callback: Callable[[DonkaCode, float],Any],  # detect_callback(which_detected, time_detected)
                 frame_left: int=1600, frame_right: int=3200,
                 in_device_idx: int = -1,
                 sample_rate: int=16000, chunk_sz=512, buffer_sz: int=15872,
                 audio_interface: pyaudio.PyAudio|None=None,
                 val_data_dir: str=".val_cache",
                 verbose: bool=False,
                 record_time_logs: bool=False,
                ):
        # Make computations easier
        assert buffer_sz % chunk_sz == 0
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
        self.reset_timing(time.time())
        
        self.audio_interface: pyaudio.PyAudio = audio_interface if audio_interface else pyaudio.PyAudio()
        self.in_device_idx = self.audio_interface.get_default_input_device_info()['index'] if in_device_idx < 0 else in_device_idx

        # Real-time operations. These are updated on every read from the audio stream
        self.rms_frame_length, self.rms_hop_length=self.chunk_sz,self.chunk_sz
        self.rms = librosa.feature.rms(y=self.audio_buffer, frame_length=self.rms_frame_length, hop_length=self.rms_hop_length, center=False)[0,]
        self.mel = librosa.feature.melspectrogram(y=self.audio_buffer, sr=sample_rate)

        self.last_10_note_sample: np.ndarray = np.zeros((10, )) - 1e9
        self.last_10_note_min_energy: np.ndarray = np.zeros((10, )) - 1

        self.start_time: float = 0

        # Operation timing
        self.record_time_logs = record_time_logs
        self.onset_time_logs = []
        self.this_classify_time = 0
        self.classify_time_logs = []
        self.add_chunk_time_logs = []
        self.loop_time_logs = []

    def reset_timing(self, time: float):
        self.start_time = time
        self.time_recv[-1] = self.start_time
        for i in range(self.buffer_sz//self.chunk_sz - 2, -1, -1):
            self.time_recv[i] = self.time_recv[i+1] - self.chunk_sz / self.sample_rate

    def detect_note(self) -> List[Tuple[DonkaCode,int]]:
        t = time.time()
        onsets = librosa.onset.onset_detect(y=self.audio_buffer, 
                                            onset_envelope=librosa.onset.onset_strength(
                                                y=self.audio_buffer,
                                                sr=self.sample_rate,
                                                S=self.mel,
                                            ),
                                            sr=self.sample_rate, 
                                            units="samples")
        #onsets = librosa.onset.onset_detect(y=self.audio_buffer, sr=self.sample_rate, units="samples")
        if self.record_time_logs: self.onset_time_logs.append(time.time() - t)

        result = []
        for onset in onsets:
            # Not enough data
            if onset < self.buffer_sz//2 or onset >= self.buffer_sz - self.frame_right:
                continue
            # Noise
            if self.rms[onset//self.rms_hop_length] <= self.noise_stat.get_energy_median() + 3*self.noise_stat.get_energy_sigma():
                continue
            
            if onset <= self.last_10_note_sample[-1] + self.frame_left:
                continue

            t = time.time()
            note = self.audio_buffer[onset-self.frame_left:onset+self.frame_right]
            notes.append((self.audio_buffer.copy(),onset))
            # Classify onset
            cls2donka_code = [
                DON|RIGHT,
                DON|LEFT,
                KA|RIGHT,
                KA|LEFT,
            ]
            cls_distance: List[Tuple[float, int]] = []
            note_mel = compute_mel_rep(note, self.sample_rate)
            for other_note_mel,cls in zip(self.train_mel, self.train_y):
                cls = np.argmax(cls)
                cls_distance.append((np.sum((other_note_mel - note_mel)**2)**0.5, cls))

            cls_distance.sort()
            donka_code = cls2donka_code[cls_distance[0][1]]

            if self.record_time_logs: self.this_classify_time += time.time() - t

            result.append((donka_code, onset))
        
        return result
    
    def register_note(self, donka_code: DonkaCode, sample: int):
        self.last_10_note_sample[:-1] = self.last_10_note_sample[1:]
        self.last_10_note_sample[-1] = sample
        
    def add_chunk(self, chunk: np.ndarray):
        assert chunk.shape == (self.chunk_sz,)

        # Shift buffers
        self.audio_buffer[:-self.chunk_sz] = self.audio_buffer[self.chunk_sz:]
        self.rms[:-1] = self.rms[1:]
        self.mel[:,:-1] = self.mel[:,1:]
        
        # Add chunk
        self.audio_buffer[-self.chunk_sz:] = chunk
        chunk_energy = librosa.feature.rms(y=self.audio_buffer[-self.chunk_sz:],
                                           frame_length=self.rms_frame_length,
                                           hop_length=self.rms_hop_length, center=False)[0,0]
        self.rms[-1] = chunk_energy
        chunk_mel = np.abs(librosa.feature.melspectrogram(y=self.audio_buffer[-3072:], sr=self.sample_rate)[:,3:])
        chunk_mel = librosa.core.power_to_db(chunk_mel)
        #print(chunk_mel.shape)
        #print(self.mel[:,-chunk_mel.shape[1]:].shape)
        self.mel[:,-chunk_mel.shape[1]:] = chunk_mel
        self.last_10_note_min_energy[-1] = min(self.last_10_note_min_energy[-1], self.rms[-1])

        # Timing
        self.last_10_note_sample -= self.chunk_sz
        self.time_recv += self.chunk_sz/self.sample_rate

    def run(self):
        in_stream = self.audio_interface.open(
            format=pyaudio.paFloat32, channels=1,
            rate=self.sample_rate, input=True,
            input_device_index=self.in_device_idx,
            frames_per_buffer=self.chunk_sz)
        self._log(f"Using input device {self.audio_interface.get_device_info_by_host_api_device_index(0, self.in_device_idx).get('name')}.",
                  tag="INFO", method="__init__")
        
        self.reset_timing(time.time())

        while self.running:
            chunk = np.frombuffer(in_stream.read(self.chunk_sz), dtype=np.float32)
            self.this_classify_time = 0
            add_chunk_start = time.time()
            self.add_chunk(chunk)
            if self.record_time_logs:
                self.add_chunk_time_logs.append(time.time() - add_chunk_start)

            for donka_code, onset in self.detect_note():
                self.register_note(donka_code, onset)

                # Note time as a linear transformation
                # In practice, the third operation here does not have effect as onset detection is chunk-based.
                note_time = self.time_recv[onset // self.chunk_sz]
                note_time -= self.chunk_sz / self.sample_rate
                note_time += (onset % self.chunk_sz) / self.sample_rate
                
                self.detect_callback(donka_code, note_time)
            
            if self.record_time_logs:
                self.loop_time_logs.append(time.time() - add_chunk_start)
                self.classify_time_logs.append(self.this_classify_time)


        in_stream.stop_stream()
        in_stream.close()

    def start(self) -> threading.Thread:
        self.running = True
        self.audio_buffer = np.zeros((self.buffer_sz,))
        self.rms = librosa.feature.rms(y=self.audio_buffer, frame_length=self.rms_frame_length, hop_length=self.rms_hop_length, center=False)[0,]
        self.last_10_note_sample: np.ndarray = np.zeros((10,)) - 1e9
        self.last_10_note_min_energy: np.ndarray = np.zeros((10,)) - 1

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
    detector = AudioInputDetector(train_x, train_y, noise_stat, lambda donka_code,t : print(donka_code, t, time.time() - t), record_time_logs=True)

    thread = detector.start()
    input()

    detector.stop()
    plt.plot(detector.loop_time_logs, label="Total Loop Time")
    plt.plot(detector.onset_time_logs, label="Onset Detect Time")
    plt.plot(detector.classify_time_logs, label="Classify Time")
    plt.plot(detector.add_chunk_time_logs, label="Add Chunk Time")
    plt.legend()
    plt.show()

    thread.join()

if __name__=="__main__":
    main()
    for note in notes:
        plt.plot(note[0])
        plt.plot([note[1]], [0], "ro")
        plt.show()