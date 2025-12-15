import os

import keras.losses
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LSTM, Input
from tensorflow.keras.optimizers import Adam
import pickle

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_classnames():
    with open("batches.meta", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    class_names = [x.decode('ascii') for x in data[b'label_names']]
    return class_names


def get_dataset():
    x_train, y_train = [], []

    for i in range(1, 6):
        with open(f"data_batch_{i}", 'rb') as fo:
            batch_dict = pickle.load(fo, encoding='bytes')
        if i == 1:
            x_train = batch_dict[b'data']
            y_train = batch_dict[b'labels']
        else:
            x_train = np.vstack((x_train, batch_dict[b'data']))
            y_train.extend(batch_dict[b'labels'])

    with open("test_batch", 'rb') as fo:
        test_batch = pickle.load(fo, encoding='bytes')
    x_test = test_batch[b'data']
    y_test = test_batch[b'labels']

    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return ((x_train, np.array(y_train)), (x_test, np.array(y_test)))


print("Загрузка данных CIFAR-10...")
(x_train, y_train), (x_test, y_test) = get_dataset()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
# y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
# class_names = get_classnames()

print(f"Данные загружены: {x_train.shape[0]} тренировочных, {x_test.shape[0]} тестовых")

callbacks = [keras.callbacks.EarlyStopping(
    monitor='accuracy',
    min_delta=0.001,
    patience=3,
    verbose=1,
    mode='max',
    restore_best_weights=True
),
    keras.callbacks.ReduceLROnPlateau(
    monitor='accuracy',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1,
    mode='max'
)]

# def create_cnn_model():
#     model = Sequential([
#         Input((32, 32, 3)),
#
#         Conv2D(32, 3, activation='relu', padding='same'),
#         BatchNormalization(),
#         Conv2D(32, 3, activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Dropout(0.25),
#
#         Conv2D(64, 3, activation='relu', padding='same'),
#         BatchNormalization(),
#         Conv2D(64, 3, activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Dropout(0.25),
#
#         Conv2D(128, 3, activation='relu', padding='same'),
#         BatchNormalization(),
#         Conv2D(128, 3, activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Dropout(0.25),
#
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(10)
#     ])
#
#     model.compile(
#         optimizer=Adam(),
#         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=['accuracy']
#     )
#
#     return model
#
#
# print("\nСоздание CNN модели...")
# cnn_model = create_cnn_model()
# cnn_model.summary()
#
# print("\nОбучение CNN модели...")
#
# cnn_history = cnn_model.fit(
#     x_train, y_train,
#     validation_data=(x_test, y_test),
#     epochs=15,
#     callbacks=callbacks,
#     batch_size=64,
#     verbose=1
# )
#
# # cnn_model = tf.keras.models.load_model('cnn_model.keras')
#
# print(f"\nИсходная CNN модель:")
# cnn_loss, cnn_accuracy = cnn_model.evaluate(x_test, y_test)
#
# # Итерационный прунинг (5 итераций)
#
# def prune_model_weights(model, pruning_rate=0.2):
#     for layer in model.layers:
#         if isinstance(layer, (Conv2D, Dense)):
#             weights = layer.get_weights()
#             if len(weights) > 0:
#                 kernel = weights[0].copy()
#                 biases = weights[1] if len(weights) > 1 else None
#
#                 flat_kernel = np.abs(kernel.flatten())
#                 threshold = np.percentile(flat_kernel, pruning_rate * 100)
#
#                 mask = np.abs(kernel) > threshold
#                 pruned_kernel = kernel * mask
#
#                 if biases is not None:
#                     layer.set_weights([pruned_kernel, biases])
#                 else:
#                     layer.set_weights([pruned_kernel])
#
#     return model
#
# print("\n" + "=" * 80)
# print("Начало итерационного прунинга (5 итераций)")
#
# original_weights = cnn_model.get_weights()
#
# pruning_rates = [0.3, 0.4, 0.5, 0.6, 0.7]
#
# for i, pruning_rate in enumerate(pruning_rates):
#     print(f"\nИтерация {i + 1}/5")
#     print(f"Прунинг {pruning_rate * 100:.0f}% наименее важных весов")
#
#     cnn_model = prune_model_weights(cnn_model, pruning_rate)
#
#     if i < 4:
#         epochs = 1
#         print(f"Fine-tuning на {epochs} эпоху...")
#     else:
#         epochs = 5
#         print(f"Fine-tuning на {epochs} эпох...")
#
#     cnn_model.compile(
#         optimizer=Adam(),
#         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=['accuracy']
#     )
#
#     history = cnn_model.fit(
#         x_train, y_train,
#         validation_data=(x_test, y_test),
#         epochs=epochs,
#         callbacks=callbacks,
#         batch_size=64,
#         verbose=1
#     )
#
#     loss, accuracy = cnn_model.evaluate(x_test, y_test)
#
#     print(f"Точность после итерации {i + 1}: {accuracy:.4f}")

# cnn_model.save('pruned_cnn_model.keras')

cnn_model = tf.keras.models.load_model('pruned_cnn_model.keras')

print("\n" + "=" * 80)
print("Дистилляция на LSTM сеть. Подбор минимальных размеров LSTM и Dense")

x_train_lstm = x_train.reshape(-1, 32, 96)
x_test_lstm = x_test.reshape(-1, 32, 96)

def create_lstm_model(lstm_units, dense_units):
    model = Sequential([
        Input((32, 96)),
        LSTM(lstm_units),
        Dense(dense_units, activation='relu'),
        Dense(10)
    ])

    return model

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False):
        batch_size = tf.shape(x)[0]
        x_reshaped = tf.reshape(x, (batch_size, 32, 32, 3))
        teacher_pred = self.teacher(x_reshaped, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            keras.ops.softmax(teacher_pred / self.temperature, axis=1),
            keras.ops.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        return self.student(x)

# Подбор размеров

_, cnn_accuracy = cnn_model.evaluate(x_test, y_test, verbose = 0)
target_accuracy = cnn_accuracy - 0.4
print(f"Допустимая точность: {target_accuracy:.4f}")
print(f"Точность пруненой CNN: {cnn_accuracy:.4f}")

lstm_sizes = [32, 64, 128, 256]
dense_sizes = [32, 64, 128, 256]
results = []

for size in lstm_sizes:
    for dense_size in dense_sizes:
        print(f"\nТестирование LSTM с {size} нейронами в LSTM и {dense_size} нейронами в Dense...")

        lstm_model = create_lstm_model(size, dense_size)

        distiller = Distiller(student=lstm_model, teacher=cnn_model)
        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.3,
            temperature=10,
        )

        distiller.fit(x_train_lstm, y_train, epochs=5, callbacks=callbacks)

        lstm_loss, lstm_accuracy = distiller.evaluate(x_test_lstm, y_test)

        results.append({
            'lstm_size': size,
            'dense_size': dense_size,
            'accuracy': lstm_accuracy,
            'params': lstm_model.count_params()
        })

print("\n" + "=" * 80)
print("Результаты подбора размеров LSTM и Dense:")

results.sort(key=lambda x: x['lstm_size'])

for result in results:
    print(f"LSTM {result['lstm_size']} нейронов и Dense {result['dense_size']}: "
          f"точность = {result['accuracy']:.4f}, "
          f"параметров = {result['params']}")

results = [x for x in results if x['accuracy'] > target_accuracy]

if len(results) > 0:
    best_result = min(results, key=lambda x: x['lstm_size'] + x['dense_size'])

    print(f"\nМинимальный подходящий размер LSTM и Dense: {best_result['lstm_size']}, {best_result['dense_size']} нейронов")
    print(f"Точность: {best_result['accuracy']:.4f}")
    print(f"Количество параметров: {best_result['params']:,}")

    best_lstm_model = create_lstm_model(best_result['lstm_size'], best_result['dense_size'])
    best_distiller = Distiller(best_lstm_model, cnn_model)
    best_distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=5,
    )
    best_distiller.fit(x_train_lstm, y_train, epochs=15, callbacks=callbacks)
    lstm_loss, lstm_accuracy = best_distiller.evaluate(x_test_lstm, y_test)
    best_lstm_model.save('distilled_lstm_model.keras')

else:
    print("Не было найдено подходящих")