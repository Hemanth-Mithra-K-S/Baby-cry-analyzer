import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score

# =========================
# LOAD TEST DATA
# =========================
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Test shape:", X_test.shape)

# =========================
# PREPARE INPUTS
# =========================

# CNN / ResNet need channel
X_cnn = X_test[..., np.newaxis]

# Transformer uses original
X_test_torch = np.transpose(X_test, (0, 2, 1))
X_test_torch = torch.tensor(X_test_torch, dtype=torch.float32)
X_trans = X_test

# =========================
# LOAD CNN MODEL
# =========================
cnn_model = tf.keras.models.load_model("best_cnn_rnn_model.h5")

# =========================
# LOAD RESNET MODEL
# =========================
resnet_model = tf.keras.models.load_model("resnet_model.h5")
# =========================
# POSITIONAL ENCODING (SAME AS TRAIN)
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
# TRANSFORMER MODEL (EXACT COPY)
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
# LOAD TRANSFORMER
# =========================
device = torch.device("cpu")

transformer = CryTransformer(input_dim=60, num_classes=5).to(device)

transformer.load_state_dict(
    torch.load("best_transformer.pth", map_location=device)
)

transformer.eval()
# =========================
# PREDICTIONS
# =========================

# CNN predictions
cnn_pred = cnn_model.predict(X_cnn, verbose=0)

# ResNet predictions
resnet_pred = resnet_model.predict(X_cnn, verbose=0)

# Transformer predictions
with torch.no_grad():
    outputs = transformer(X_test_torch)
    transformer_probs = torch.softmax(outputs, dim=1).numpy()

# =========================
# ENSEMBLE (WEIGHTED)
# =========================

# weights (you can tune this)
w1 = 0.35   # CNN
w2 = 0.50   # ResNet
w3 = 0.10   # Transformer


final_pred = (w1 * cnn_pred) + (w2 * resnet_pred) + (w3 * transformer_probs)

# =========================
# FINAL LABELS
# =========================
y_pred = np.argmax(final_pred, axis=1)

# =========================
# ACCURACY
# =========================
acc = accuracy_score(y_test, y_pred)

print("\n🔥 ENSEMBLE ACCURACY:", round(acc * 100, 2), "%")