import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# =========================
# LOAD DATA
# =========================
X = np.load("X.npy")
y = np.load("y.npy")

print("Loaded Shape:", X.shape)

# ADD CHANNEL
X = X[..., np.newaxis]
print("Final Shape:", X.shape)

# =========================
# ENCODE LABELS
# =========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# SAVE LABELS
np.save("labels.npy", le.classes_)

# =========================
# CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights = dict(enumerate(class_weights))


# =========================
# MODEL
# =========================
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # CNN
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    # RESHAPE
    x = Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)

    # RNN
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    x = LSTM(64)(x)
    x = Dropout(0.3)(x)

    # DENSE
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =========================
# TRAIN
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

input_shape = X.shape[1:]
num_classes = y_cat.shape[1]

accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n🔥 Fold {fold}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_cat[train_idx], y_cat[val_idx]

    model = build_model(input_shape, num_classes)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=16,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=6, restore_best_weights=True),
            ReduceLROnPlateau(patience=3)
        ]
    )

    loss, acc = model.evaluate(X_val, y_val)
    print("Accuracy:", acc)
    accuracies.append(acc)

print("\n✅ Average Accuracy:", np.mean(accuracies))

model.save("best_cnn_rnn_model.h5")
print("✅ Model saved!")