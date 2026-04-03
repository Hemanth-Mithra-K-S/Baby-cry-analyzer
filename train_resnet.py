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
print("Classes:", le.classes_)
np.save("class_names.npy", le.classes_)

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
# RESNET BLOCK
# =========================
def res_block(x, filters):
    shortcut = x

    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    # Match dimensions
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1,1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


# =========================
# MODEL
# =========================
def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    # RES BLOCKS
    x = res_block(x, 32)
    x = MaxPooling2D()(x)

    x = res_block(x, 64)
    x = MaxPooling2D()(x)

    x = res_block(x, 128)
    x = MaxPooling2D()(x)

    x = res_block(x, 256)
    x = MaxPooling2D()(x)

    # GLOBAL POOL
    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
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

    model = build_resnet(input_shape, num_classes)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=7, restore_best_weights=True),
            ReduceLROnPlateau(patience=3)
        ]
    )

    loss, acc = model.evaluate(X_val, y_val)
    print("Accuracy:", acc)
    accuracies.append(acc)

print("\n✅ Average Accuracy:", np.mean(accuracies))

model.save("resnet_model.h5")
print("✅ ResNet model saved!")