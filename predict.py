import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import librosa

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 22050
DURATION = 4
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 60
MAX_LEN = 200

# =========================
# LOAD CLASS NAMES
# =========================
class_names = np.load("class_names.npy", allow_pickle=True)

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if len(signal) < SAMPLES_PER_TRACK:
        signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)))
    else:
        signal = signal[:SAMPLES_PER_TRACK]

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0,0),(0, MAX_LEN - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc


# =========================
# TRANSFORMER MODEL
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        pe = torch.zeros(500, d_model)
        position = torch.arange(0, 500).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:,:x.size(1)].to(x.device)


class CryTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pos_encoder = PositionalEncoding(128)

        encoder = nn.TransformerEncoderLayer(128,4,256,0.3,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=4)

        self.attention_pool = nn.Linear(128,1)

        self.fc = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64,num_classes)
        )

    def forward(self,x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        w = torch.softmax(self.attention_pool(x),dim=1)
        x = torch.sum(w*x,dim=1)

        return self.fc(x)


# =========================
# LOAD MODELS
# =========================
device = torch.device("cpu")

# CNN-RNN
cnn_model = tf.keras.models.load_model("best_cnn_rnn_model.h5")

# ResNet
resnet_model = tf.keras.models.load_model("resnet_model.h5")

# Transformer
transformer = CryTransformer(60, len(class_names)).to(device)
transformer.load_state_dict(torch.load("best_transformer.pth", map_location=device))
transformer.eval()

print("✅ All models loaded!")

# =========================
# MAIN PREDICT FUNCTION
# =========================
def predict_audio(file_path):

    mfcc = extract_features(file_path)

    # ---------- CNN / ResNet ----------
    X_cnn = mfcc[..., np.newaxis]   # (60,200,1)
    X_cnn = np.expand_dims(X_cnn, axis=0)

    cnn_pred = cnn_model.predict(X_cnn, verbose=0)
    resnet_pred = resnet_model.predict(X_cnn, verbose=0)

    # ---------- Transformer ----------
    X_trans = (mfcc - np.mean(mfcc)) / (np.std(mfcc)+1e-6)
    X_trans = np.transpose(X_trans, (1,0))   # (200,60)
    X_trans = np.expand_dims(X_trans, axis=0)

    X_tensor = torch.tensor(X_trans, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = transformer(X_tensor)
        transformer_pred = torch.softmax(out, dim=1).cpu().numpy()

    # =========================
    # ENSEMBLE (WEIGHTS)
    # =========================
    w1, w2, w3 = 0.35, 0.50, 0.15

    final_pred = (w1 * cnn_pred) + (w2 * resnet_pred) + (w3 * transformer_pred)

    pred = np.argmax(final_pred)
    confidence = np.max(final_pred)

    return class_names[pred], float(confidence)