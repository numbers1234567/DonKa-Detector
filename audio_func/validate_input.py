from typing import Tuple,List,Dict
import numpy as np
import random
from Levenshtein import distance as levenshtein_distance
import librosa.onset
import librosa
import os
import os.path
import struct
import matplotlib.pyplot as plt

try:
    from utility import compute_mel_rep,retrieve_audio_inputs,AudioStatistics
except:
    from .utility import compute_mel_rep,retrieve_audio_inputs,AudioStatistics

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

def retrieve_chart_multik(audio: np.ndarray, sr: int, train_x: List[np.ndarray], train_y: np.ndarray, frame_left: int=1600, frame_right: int=3200, normalize=True) -> List[Tuple[int, int]]:
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
    train_mel: List[np.ndarray] = [compute_mel_rep(note, sr) for note in train_x]
    if normalize: train_mel: List[np.ndarray] = [i/np.sum(i) for i in train_mel]

    last_onset = -frame_left

    answer = []

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

        distances = []
        for other,cls in zip(train_mel, train_y):
            cls = np.argmax(cls)
            distances.append((np.sqrt(np.sum((note_mel - other) ** 2)), cls))

        distances.sort()
        answer.append((onset, [c for d,c in distances]))

        last_onset = onset
    return answer

type KNNHyperParamK = int
type KNNHyperParamIsNorm = bool
type KNNHyperParam = Tuple[KNNHyperParamK, KNNHyperParamIsNorm]
type KNNHyperParamMetric = Dict[KNNHyperParam, Dict[float, Tuple[float, float]]]
def evaluate(normalize: bool, K_max: int, 
             train_x: List[np.ndarray], train_y: np.ndarray, 
             test_x: List[np.ndarray], test_y: np.ndarray, 
             experiments: List[Tuple[int,float]], 
             sr: int, noise_stat: AudioStatistics, 
             verbose=True, seed: int=42) -> KNNHyperParamMetric:
    # So datasets are the same per-evaluation
    np.random.seed(seed)
    random.seed(seed)


    hist = [[] for _ in range(K_max)]

    # Metrics over all experiments
    result: KNNHyperParamMetric = {}
    if verbose: print(f"==== normalize={normalize} ====")
    for samples,note_rate in experiments:
        # Experiment-specific metric
        num_pred = [0] * K_max
        base_l_distance = [0] * K_max
        bin_l_distance = [0] * K_max

        print(f"samples={samples}, rate={note_rate}")

        pbar = range(samples)
        for i in pbar:
            # Create test data
            audio, chart_gt = create_random_chart(test_x, test_y, note_rate=note_rate, sr=sr, noise_std=noise_stat.get_wav_std())
            chart_gt_sparse = [np.argmax(i) for idx,i in enumerate(chart_gt) if np.max(i) > 0.5]
            
            # Predict chart
            chart_pred = retrieve_chart_multik(audio, sr, train_x, train_y, normalize=normalize)            
            chart_pred = [cls for onset,cls in chart_pred]

            # Predictions for each K
            chart_pred_k = [[] for _ in range(K_max)]
            for cls in chart_pred:
                votes = [0] * 4
                for k in range(1, K_max + 1):
                    votes[cls[k-1]] += 1
                    if votes[0] + votes[1] > votes[2] + votes[3]:
                        chart_pred_k[k-1].append(np.argmax(votes[:2]))
                    elif votes[0] + votes[1] < votes[2] + votes[3]:
                        chart_pred_k[k-1].append(np.argmax(votes[2:]) + 2)
                    else:
                        chart_pred_k[k-1].append(np.argmax(votes))

            # Compute metrics
            for k in range(1, K_max + 1):
                base_l_distance[k-1] += levenshtein_distance(chart_gt_sparse, chart_pred_k[k-1])

            chart_gt_sparse = [cls//2 for cls in chart_gt_sparse]

            for k in range(1, K_max + 1):
                chart_pred_k[k-1] = [cls//2 for cls in chart_pred_k[k-1]]
                bin_l_distance[k-1] += levenshtein_distance(chart_gt_sparse, chart_pred_k[k-1])
                if (k,normalize) not in result:
                    result[(k, normalize)] = {}
                result[(k, normalize)][note_rate] = (bin_l_distance[k-1], base_l_distance[k-1])

                num_pred[k-1] += len(chart_pred_k[k-1])
        if verbose:
            for idx,(n,base_d,bin_d) in enumerate(zip(num_pred, base_l_distance, bin_l_distance), start=1):
                print(f" - K={idx}: ({n}, {base_d}, {bin_d})")
    return result

def main():
    sr=16000
    # Retrieve all data from audio/user
    note_x,note_y,noise_stat = retrieve_audio_inputs()

    train_x,train_y = note_x[::2],note_y[::2]
    test_x,test_y = note_x[1::2],note_y[1::2]


    
    hp = [
        normalize
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

    
    results: KNNHyperParamMetric = {}

    for normalize in hp:
        results.update(
            evaluate(normalize, 5, 
                     train_x, train_y, 
                     test_x, test_y, 
                     experiments, sr, noise_stat=noise_stat, 
                     verbose=True)
        )

    # Save results
    val_dir = ".val_cache"
    if not os.path.isdir(val_dir):
        os.mkdir(val_dir)

    with open(os.path.join(val_dir, "knn"), "wb") as f:
        for (k,is_norm),sub_res in results.items():
            for note_rate,(bin_acc,base_acc) in sub_res.items():
                b = struct.pack("I?ddd", k, is_norm, note_rate, bin_acc, base_acc)
                f.write(b)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="DonKa Input Validation",
        description="Validate user-provided input"
    )

    main()