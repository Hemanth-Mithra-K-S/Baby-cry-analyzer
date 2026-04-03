import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
BATCH_SIZE = 16
EPOCHS = 60
LR = 0.0003

# =========================
# LOAD DATA
# =========================
X = np.load("X.npy", allow_pickle=True)
y = np.load("y.npy", allow_pickle=True)

print("Original shape:", X.shape)

# =========================
# NORMALIZE (VERY IMPORTANT)
# =========================
X = (X - np.mean(X)) / (np.std(X) + 1e-6)

# =========================
# ENCODE LABELS
# =========================
le = LabelEncoder()
y = le.fit_transform(y)

print("Classes:", le.classes_)
np.save("class_names.npy", le.classes_)

NUM_CLASSES = len(le.classes_)

# =========================
# TRANSPOSE (IMPORTANT)
# (batch, time, features)
# =========================
X = np.transpose(X, (0, 2, 1))  # (1880, 200, 60)

# =========================
# SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# DATASET
# =========================
class CryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(CryDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CryDataset(X_val, y_val), batch_size=BATCH_SIZE)

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
# STRONG TRANSFORMER MODEL
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

        # 🔥 Attention pooling (better than mean)
        weights = torch.softmax(self.attention_pool(x), dim=1)
        x = torch.sum(weights * x, dim=1)

        return self.fc(x)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CryTransformer(input_dim=X.shape[2], num_classes=NUM_CLASSES).to(device)

# =========================
# LOSS
# =========================
class_counts = np.bincount(y)
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.5
)

# =========================
# TRAINING
# =========================
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, pred = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (pred == y_batch).sum().item()

    train_acc = 100 * correct / total

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            _, pred = torch.max(outputs, 1)

            val_total += y_batch.size(0)
            val_correct += (pred == y_batch).sum().item()

    val_acc = 100 * val_correct / val_total

    scheduler.step(val_acc)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {total_loss/len(train_loader):.4f}")
    print(f"Train Acc: {train_acc:.2f}%")
    print(f"Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_transformer.pth")
        print("✅ Best model saved!")

print("\n🔥 Best Validation Accuracy:", best_acc)