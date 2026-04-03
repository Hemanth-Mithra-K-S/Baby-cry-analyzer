import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("X.npy")
y = np.load("y.npy")

# encode labels if string
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

np.save("C:/Babycryanalyzer/final_test/X_test.npy", X_test)
np.save("C:/Babycryanalyzer/final_test/y_test.npy", y_test)

print("✅ Test data saved")