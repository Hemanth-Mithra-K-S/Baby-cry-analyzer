import numpy as np
import librosa
import tensorflow as tf
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 22050
DURATION = 4
N_MFCC = 60
MAX_LEN = 200

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODELS
# =========================
cnn_model = tf.keras.models.load_model("best_cnn_rnn_model.h5")
resnet_model = tf.keras.models.load_model("resnet_model.h5")

# =========================
# TRANSFORMER MODEL
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

class CryTransformer(nn.Module):
    def __init__(self, input_dim=60, num_classes=5):
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

device = torch.device("cpu")

transformer = CryTransformer()
transformer.load_state_dict(torch.load("best_transformer.pth", map_location=device))
transformer.eval()

# =========================
# LOAD CLASS NAMES
# =========================
class_names = np.load("class_names.npy")

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(file):
    try:
        # 🔥 Force librosa to load any format (webm/mp3/wav)
        signal, sr = librosa.load(file, sr=SAMPLE_RATE, mono=True)

        if signal is None or len(signal) == 0:
            raise ValueError("Empty or invalid audio")

        # Fix length
        if len(signal) < SAMPLE_RATE * DURATION:
            pad = SAMPLE_RATE * DURATION - len(signal)
            signal = np.pad(signal, (0, pad))
        else:
            signal = signal[:SAMPLE_RATE * DURATION]

        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)))
        else:
            mfcc = mfcc[:, :MAX_LEN]

        return mfcc

    except Exception as e:
        print("❌ Feature extraction error:", e)
        return None

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("✅ Request received")

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files['file']
        import subprocess

        # Save webm
        webm_path = "temp.webm"
        wav_path = "temp.wav"

        file.save(webm_path)

        # Convert to wav using ffmpeg
        command = f'ffmpeg -y -i "{webm_path}" "{wav_path}"'
        subprocess.run(command, shell=True)

        print("✅ Converted to WAV")
        print("✅ File saved")

        features = extract_features(wav_path)
        if features is None:
            print("❌ Feature extraction failed")
            return jsonify({"error": "Audio processing failed"})

        # CNN / ResNet
        X_cnn = features[np.newaxis, ..., np.newaxis]
        cnn_pred = cnn_model.predict(X_cnn, verbose=0)
        resnet_pred = resnet_model.predict(X_cnn, verbose=0)

        # Transformer
        X_trans = np.transpose(features, (1, 0))
        X_trans = torch.tensor(X_trans[np.newaxis, ...], dtype=torch.float32)

        with torch.no_grad():
            out = transformer(X_trans)
            trans_pred = torch.softmax(out, dim=1).numpy()

        # ENSEMBLE
        final = (0.3 * cnn_pred) + (0.4 * resnet_pred) + (0.3 * trans_pred)

        pred_class = np.argmax(final)
        label = class_names[pred_class]
        
        probabilities = {
            str(class_names[i]): float(final[0][i])
            for i in range(len(class_names))
        }

        print("✅ Prediction done:", label)

        return jsonify({
            "prediction": str(label),
            "probabilities": probabilities
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)})

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)