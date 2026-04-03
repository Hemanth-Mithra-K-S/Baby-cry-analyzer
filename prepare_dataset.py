import os
import librosa
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATASET_PATH = "C:/Babycryanalyzer/baby_app/data"   # <-- CHANGE THIS
SAMPLE_RATE = 22050
DURATION = 4
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# MFCC SETTINGS
N_MFCC = 60
MAX_LEN = 200

# =========================
# FUNCTION
# =========================
def extract_features(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        if len(signal) < SAMPLES_PER_TRACK:
            pad = SAMPLES_PER_TRACK - len(signal)
            signal = np.pad(signal, (0, pad))
        else:
            signal = signal[:SAMPLES_PER_TRACK]

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=N_MFCC
        )

        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)))
        else:
            mfcc = mfcc[:, :MAX_LEN]

        return mfcc

    except:
        return None


# =========================
# LOAD DATA
# =========================
X = []
y = []

labels = os.listdir(DATASET_PATH)

for label in labels:
    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    print(f"Processing {label}")

    for file in tqdm(os.listdir(folder)):
        file_path = os.path.join(folder, file)

        feature = extract_features(file_path)

        if feature is not None:
            X.append(feature)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Final Shape:", X.shape)

np.save("X.npy", X)
np.save("y.npy", y)

print("✅ Dataset saved successfully!")