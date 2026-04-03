import numpy as np
import librosa
import tensorflow as tf

# LOAD MODEL
model = tf.keras.models.load_model("resnet_model.h5")

# LOAD LABELS
labels = np.load("labels.npy")

# CONFIG
SAMPLE_RATE = 22050
DURATION = 4
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 60
MAX_LEN = 200


def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if len(signal) < SAMPLES_PER_TRACK:
        pad = SAMPLES_PER_TRACK - len(signal)
        signal = np.pad(signal, (0, pad))
    else:
        signal = signal[:SAMPLES_PER_TRACK]

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0,0),(0,MAX_LEN-mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc


# TEST AUDIO
file_path = "C:/Babycryanalyzer/Resnet/test_belly2.wav"   # <-- CHANGE THIS

mfcc = extract_features(file_path)
mfcc = mfcc[np.newaxis, ..., np.newaxis]

prediction = model.predict(mfcc)
predicted_class = labels[np.argmax(prediction)]

print("\n🎯 Prediction:", predicted_class)