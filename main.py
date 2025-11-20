import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import pickle

from tensorflow.python.keras.utils.np_utils import to_categorical

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

x_train = []
y_train = []

for i in range(1, 6):
    with open(f"data_batch_{i}", 'rb') as fo:
        batch_dict = pickle.load(fo, encoding='bytes')
    if i == 1:
        x_train = batch_dict[b'data']
        y_train = batch_dict[b'labels']
    else:
        x_train = np.vstack((x_train, batch_dict[b'data']))
        y_train.extend(batch_dict[b'labels'])

with open(f"test_batch", 'rb') as fo:
    test_batch = pickle.load(fo, encoding='bytes')
x_test = test_batch[b'data']
y_test = test_batch[b'labels']

x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

y_train = to_categorical(np.array(y_train), 10)
y_test = to_categorical(np.array(y_test), 10)

model = Sequential([
    Conv2D(32, 3, activation = 'relu', padding='same', input_shape = (32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, 3, activation = 'relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, 3, activation = 'relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, 3, activation = 'relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, 3, activation = 'relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, 3, activation = 'relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(256, 3, activation = 'relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, 3, activation = 'relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),

    Dense(10, activation = 'softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

model.summary()

EPOCHS = 10
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=EPOCHS)

metrics = history.history
plt.plot(history.epoch, metrics["loss"], metrics["val_loss"])
plt.legend(["loss", "val_loss"])
plt.show()