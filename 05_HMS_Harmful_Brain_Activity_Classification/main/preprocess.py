import os
import sys
import gc
from argparse import Namespace

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import pywt, librosa

if os.path.isdir("/kaggle"):
    a = 0
    ROOT = "/kaggle"
else:
    sys.path.append("../main")
    ROOT = ".."


def load_train(meta_csv_path, spec_path, eeg_path, read_spec_file, read_eeg_spec_file):
    meta_data, targets = preprocess_train_df(meta_csv_path)
    spectrograms = read_spectrogram(spec_path, read_spec_file)
    all_eegs = read_eeg(meta_data, eeg_path, read_eeg_spec_file)
    return meta_data, targets, spectrograms, all_eegs


def load_test(meta_path, spectrogram_path, eeg_path):
    meta_df, test_spectrograms = test_meta_and_spectrogram(meta_path, spectrogram_path)

    test_eeg_ids = meta_df.eeg_id.unique()
    test_eegs = all_spectrogram_from_eeg(eeg_path, test_eeg_ids)
    return meta_df, test_spectrograms, test_eegs


def preprocess_train_df(csv_path, vote_bins=4, clip_max=100):
    df = pd.read_csv(csv_path)
    targets = df.columns[-6:]
    print("Train shape:", df.shape)
    print("Targets", list(targets))
    train = df.groupby("eeg_id")[
        ["spectrogram_id", "spectrogram_label_offset_seconds"]
    ].agg({"spectrogram_id": "first", "spectrogram_label_offset_seconds": "min"})
    train.columns = ["spec_id", "min"]

    tmp = df.groupby("eeg_id")[
        ["spectrogram_id", "spectrogram_label_offset_seconds"]
    ].agg({"spectrogram_label_offset_seconds": "max"})
    train["max"] = tmp
    tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
    train["patient_id"] = tmp
    tmp = df.groupby("eeg_id")[targets].agg("sum")
    for t in targets:
        train[t] = tmp[t].values
    y_data = train[targets].values
    vote_count = y_data.sum(axis=1)
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    train[targets] = y_data
    tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
    train["target"] = tmp
    train["vote_count"] = vote_count
    train["vote_count"].clip(0, clip_max, inplace=True)
    vote_count_dict = {}
    for i in range(0, clip_max + 1):
        vote_count_dict[i] = i // (clip_max // vote_bins)
    train["vote_category"] = train["vote_count"].map(lambda x: vote_count_dict[x])
    train = train.reset_index()
    print("Train non-overlapp eeg_id shape:", train.shape)
    train.head()
    return train, targets


def read_spectrogram(spec_path, read_spec_file):
    files = os.listdir(spec_path)
    print(f"There are {len(files)} spectrogram parquets")
    if read_spec_file:
        spectrograms = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{spec_path}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
    else:
        spectrograms = np.load(
            f"{ROOT}/input/brain-spectrograms/specs.npy", allow_pickle=True
        ).item()
    return spectrograms


def read_eeg(meta_data, eeg_path, read_eeg_spec_file):
    if read_eeg_spec_file:
        all_eegs = {}
        for i, e in enumerate(meta_data.eeg_id.values):
            if i % 100 == 0:
                print(i, ", ", end="")
            x = np.load(f"{eeg_path}/{e}.npy")
            all_eegs[e] = x
    else:
        all_eegs = np.load(f"{eeg_path}eeg_specs.npy", allow_pickle=True).item()
    return all_eegs


def test_meta_and_spectrogram(test_meta_path, test_spectrogram_path):
    test_meta_df = pd.read_csv(test_meta_path)
    print("Test shape", test_meta_df.shape)
    test_meta_df.head()
    PATH2 = test_spectrogram_path
    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test spectrogram parquets")

    test_spectrograms = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])
        test_spectrograms[name] = tmp.iloc[:, 1:].values

    # RENAME FOR DATALOADER
    test_meta_df = test_meta_df.rename({"spectrogram_id": "spec_id"}, axis=1)
    return test_meta_df, test_spectrograms


USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])

    ret = pywt.waverec(coeff, wavelet, mode="per")

    return ret


def spectrogram_from_eeg(parquet_path, display=False):

    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128, 256, 16), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 256,
                n_fft=1024,
                n_mels=128,
                fmin=0,
                fmax=20,
                win_length=128,
            )

            # LOG TRANSFORM
            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]

            # STANDARDIZE TO -1 TO 17890
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k * 4 + kk] = mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        # img[:, :, k] /= 4.0

    return img


def all_spectrogram_from_eeg(eeg_path, eeg_ids):
    all_eegs = {}
    for i, eeg_id in enumerate(eeg_ids):
        img = spectrogram_from_eeg(f"{eeg_path}{eeg_id}.parquet", False)
        all_eegs[eeg_id] = img
    return all_eegs
