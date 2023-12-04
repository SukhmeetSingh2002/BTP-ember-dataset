import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# ⛔️ training data has non-float values
data = [
    [0.1, 0.2, '0.3'],
    [0.1, 0.2, '0.3'],
    [0.1, 0.2, '0.3'],
]

X_train = pd.DataFrame(data=data, columns=["x1", "x2", "x3"])
y_train = pd.DataFrame(data=[1, 0, 1], columns=["y"])


print(X_train)

print(X_train.dtypes)


model = Sequential()
model.add(
    Dense(1, input_dim=X_train.shape[1], activation='sigmoid')
)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ⛔️ ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).
model.fit(X_train.to_numpy(), y_train, epochs=3)
