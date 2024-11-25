from typing import Tuple,List,Dict
import numpy as np
import random
from Levenshtein import distance as levenshtein_distance
import librosa.onset
import librosa
from sklearn.neighbors import KNeighborsClassifier
import os
import matplotlib.pyplot as plt

DON = 0
KA  = 1

LEFT  = 0
RIGHT = 2

def create_random_chart(x: List[np.ndarray], y: np.ndarray, noise_std: float=0.01, note_rate: float=2, sr:int=16000, time_s: float = 5, frame_left: int=1600, frame_right: int=3200) -> Tuple[np.ndarray, np.ndarray]:
    # x: Arrays are wave forms
    # y: (len(x), 4): Class of each wave form in x
    # noise_std: Standard deviation of white noise added to signal
    # note_rate: notes per second
    # sr: sample rate
    # time_s: length of chart in seconds
    
    assert len(x) == len(y)
    assert y.shape==(len(x), 4)
    assert False not in [len(i.shape) == 1 for i in x]

    total_samples = int(sr*time_s)

    audio = np.random.normal(scale=noise_std,size=(total_samples,))
    chart = np.zeros((total_samples,4))

    samples_bw = int(sr / note_rate)

    # Get the waveforms for each class
    waveforms_class: List[List[np.ndarray]] = [[],[],[],[]]
    for donka,cls in zip(x,y):
        cls = np.argmax(cls)
        waveforms_class[cls].append(donka)

    # Add waveforms randomly, but on-beat. Use first second as background noise
    for i in range(sr, total_samples, samples_bw):
        note = np.random.randint(0,5)
        if note == 4:
            continue

        select_waveform = random.choice(waveforms_class[note])
        if i + len(select_waveform) >= total_samples:
            break
        chart[i][note] = 1
        audio[i-frame_left:i+frame_right] += select_waveform
    

    return audio,chart

def compute_mel_rep(note: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.melspectrogram(y=note, sr=sr, n_mels=128, fmax=8000)

def retrieve_chart_multi_k(audio: np.ndarray, sr: int, train_x: List[np.ndarray], train_y: np.ndarray, frame_left: int=1600, frame_right: int=3200, K=5, normalize=True) -> List[Tuple[int, int]]:
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, units="samples")

    random.seed(123)
    train_x = train_x.copy()

    # Calculate RMS energy per frame.
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=512)[0,]
    envtm = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    # Use first second as a representative of background noise
    noiseidx = envtm <= 1
    noisemedian = np.percentile(rms[noiseidx], 50)
    sigma = np.percentile(rms[noiseidx], 84.1) - noisemedian

    threshold = noisemedian + 3*sigma
    
    # Transform training data
    train_mel: List[np.ndarray] = [compute_mel_rep(note, sr).flatten() for note in train_x]
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

        note_mel = compute_mel_rep(note, sr=sr)
        if normalize: note_mel /= np.sum(note_mel)

        # Problem-specific vote system
        votes = knn.predict_proba([note_mel.flatten()])[0]
        if votes[0] + votes[1] > votes[2] + votes[3]:
            answer.append((onset, np.argmax(votes[:2])))
        else:
            answer.append((onset, np.argmax(votes[2:4]) + 2))
        last_onset = onset
    return answer

def evaluate(normalize: bool, K: int, train_x: List[np.ndarray], train_y: np.ndarray, test_x: List[np.ndarray], test_y: np.ndarray, experiments: List[Tuple[int,float]], sr: int, noise_std: float=0.01, verbose=True, seed: int=42):
    # So datasets are the same per-evaluation
    np.random.seed(seed)
    random.seed(seed)


    hist = []

    # Metrics over all experiments
    total_base_distance = 0
    total_bin_distance = 0
    total_pred = 0
    if verbose: print(f"==== K={K} normalize={normalize} ====")
    for samples,note_rate in experiments:
        # Experiment-specific metric
        num_pred = 0
        base_l_distance = 0
        bin_l_distance = 0

        pbar = range(samples)
        for i in pbar:
            # Create test data
            audio, chart_gt = create_random_chart(test_x, test_y, note_rate=note_rate, sr=sr, noise_std=noise_std)
            chart_gt_sparse = [np.argmax(i) for idx,i in enumerate(chart_gt) if np.max(i) > 0.5]
            
            # Predict chart
            chart_pred = retrieve_chart(audio, sr, train_x, train_y, K=K, normalize=normalize)            
            chart_pred = [cls for onset,cls in chart_pred]

            # Compute metrics
            base_l_distance += levenshtein_distance(chart_gt_sparse, chart_pred)

            chart_pred = [cls//2 for cls in chart_pred]
            chart_gt_sparse = [cls//2 for cls in chart_gt_sparse]
            bin_l_distance += levenshtein_distance(chart_gt_sparse, chart_pred)

            num_pred += len(chart_pred)

        hist.append((base_l_distance, bin_l_distance, num_pred))
        if verbose:
            print(f"Experiment ({samples}, {note_rate}): ({num_pred}, {base_l_distance}, {bin_l_distance})")

        total_base_distance += base_l_distance
        total_bin_distance += bin_l_distance
        total_pred += num_pred
    return (total_base_distance, total_bin_distance, total_pred),hist

def retrieve_audio_inputs(target_dir: str="audio/user", sr: int=16000) -> Tuple[List[np.ndarray], np.ndarray, float]:
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
    noise_std: float = np.std(noise_arr)

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

    return note_x,note_y,noise_std

def main():
    sr=16000
    # Retrieve all data from audio/user
    note_x,note_y,noise_std = retrieve_audio_inputs()

    train_x,train_y = note_x[::2],note_y[::2]
    test_x,test_y = note_x[1::2],note_y[1::2]


    
    hp = [
        [normalize, k]
        for k in range(1,6)
        for normalize in (True, False)
    ]
    experiments: List[Tuple[int,float]] = [ # Number of examples, note_rate
        (10, 1),
        (10, 1.5),
        (10, 2),
        (10, 2.5),
        (10, 3),
        (10, 4),
        (10, 5),
        (10, 6),
        (10, 7),
        (10, 8),
    ]

    for normalize,k in hp:
        evaluate(normalize, k, train_x, train_y, test_x, test_y, experiments, sr, noise_std=noise_std, verbose=True)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="DonKa Input Validation",
        description="Validate user-provided input"
    )

    main()