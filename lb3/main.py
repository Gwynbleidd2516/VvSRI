import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import pickle

from tensorflow.python.keras.utils.np_utils import to_categorical

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def predict_single_image(model, image, class_names):

    prediction = model.predict(image, verbose=1)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    return (class_names[predicted_class], confidence, prediction[0])

def get_classnames():
    with open(f"batches.meta", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    class_names = [x.decode('ascii') for x in data[b'label_names']]
    return class_names

def get_dataset():
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

    return ((x_train, y_train), (x_test, y_test))

(x_train, y_train), (x_test, y_test) = get_dataset()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

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

LAMBDA = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LAMBDA),
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

print("\n" + '='*100 + '\nОбучение нейросети')

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
]

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=7, callbacks=callbacks, batch_size=32)
model.save('my_model.keras')

print("\n" + '='*100 + '\nТочность нейросети и ошибка')

metrics = history.history
plt.plot(history.epoch, metrics["loss"], metrics["val_loss"])
plt.legend(["loss", "val_loss"])
plt.show()

#model = tf.keras.models.load_model('MODEL.keras')

loss, accuracy = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

print("\n" + '='*100 + '\nПримеры ответов обученной нейросети')

(x_train, y_train), (x_test, y_test) = get_dataset()

class_names = get_classnames()

correct = []
incorrect = []

for i, x in enumerate(x_train):
    image_batch = np.expand_dims(x, axis=0)
    prediction = model.predict(image_batch, verbose=0)
    predicted_class = class_names[np.argmax(prediction[0])]
    if predicted_class != class_names[y_train[i]] and len(incorrect) < 4:
        incorrect.append((x, y_train[i]))
    elif len(correct) < 4:
        correct.append((x, y_train[i]))

    if len(correct) == 4 and len(incorrect) == 4: break

plt.figure(figsize=(12,8))

for i in range(8):
    plt.subplot(2,4,i+1)

    image_arr = correct
    j = i
    if i > 3:
        j = i - 4
        image_arr = incorrect

    plt.imshow(image_arr[j][0], interpolation='none')
    image_batch = np.expand_dims(image_arr[j][0], axis=0)
    prediction = model.predict(image_batch, verbose=0)
    predicted_class = class_names[np.argmax(prediction[0])]
    plt.title(f"True class: {class_names[image_arr[j][1]]},\npredicted class: {predicted_class}")
plt.tight_layout()
plt.show()