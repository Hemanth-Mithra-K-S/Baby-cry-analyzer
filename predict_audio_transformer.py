import numpy as np
import torch
import torch.nn as nn
import librosa

# =========================
# CONFIG (MUST MATCH TRAIN)
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

    # Fix length
    if len(signal) < SAMPLES_PER_TRACK:
        pad = SAMPLES_PER_TRACK - len(signal)
        signal = np.pad(signal, (0, pad))
    else:
        signal = signal[:SAMPLES_PER_TRACK]

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Fix time dimension
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc

# =========================
# POSITIONAL ENCODING
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# =========================
# TRANSFORMER MODEL
# =========================
class CryTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.pos_encoder = PositionalEncoding(128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.attention_pool = nn.Linear(128, 1)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)

        x = self.transformer(x)

        weights = torch.softmax(self.attention_pool(x), dim=1)
        x = torch.sum(weights * x, dim=1)

        return self.fc(x)

# =========================
# LOAD MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CryTransformer(input_dim=60, num_classes=len(class_names)).to(device)

model.load_state_dict(
    torch.load("best_transformer.pth", map_location=device)
)

model.eval()

print("✅ Transformer loaded!")

# =========================
# GIVE AUDIO PATH HERE
# =========================
audio_path = "C:/Babycryanalyzer/Resnet/test_burp3.wav"   # 🔥 CHANGE THIS

# =========================
# PROCESS AUDIO
# =========================
features = extract_features(audio_path)

# Normalize (same as training)
features = (features - np.mean(features)) / (np.std(features) + 1e-6)

# Reshape for model
features = np.transpose(features, (1, 0))   # (200, 60)
features = np.expand_dims(features, axis=0) # (1, 200, 60)

X_tensor = torch.tensor(features, dtype=torch.float32).to(device)

# =========================
# PREDICT
# =========================
with torch.no_grad():
    outputs = model(X_tensor)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()

pred_class = np.argmax(probs)
confidence = np.max(probs)

# =========================
# OUTPUT
# =========================
print("\n🎯 Prediction Result:")
print("Predicted Class:", class_names[pred_class])
print("Confidence:", round(confidence * 100, 2), "%")