"""
This script is responsible for building the different
models used to test numerical errors during model evaluation.
"""

import tensorflow as tf
from tensorflow import keras

def load_data(dataset):
    if dataset == "mnist":
        return keras.datasets.mnist.load_data()
    if dataset == "fashion_mnist":
        return keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def prep_data(dataset, norm=False):
    (x_train, y_train), (x_test, y_test) = load_data(dataset)

    x_train, x_val = x_train[:-5000], x_train[-5000:]
    y_train, y_val = y_train[:-5000], y_train[-5000:]

    if norm:
        x_train = x_train / 255.0
        x_val = x_val / 255.0
        x_test = x_test / 255.0

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_model(dtype="float64"):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), dtype=dtype),
        tf.keras.layers.Dense(128, activation='relu', dtype=dtype),
        tf.keras.layers.Dense(64, activation='relu', dtype=dtype),
        tf.keras.layers.Dense(10, activation='softmax', dtype=dtype)
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model


def train_and_save(dataset, name, norm=False, dtype="float64", epoch=10, batch_size=32):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prep_data(dataset, norm)

    model = build_model(dtype)

    model.fit(x_train, y_train, epochs=epoch, verbose=1, batch_size=batch_size, validation_data=(x_val, y_val))

    model.evaluate(x_test, y_test)
    model.save(name)



train_and_save("mnist","models/mnist_model_f64.keras")
train_and_save("mnist","models/mnist_model_norm.keras",True)
train_and_save("mnist","models/mnist_model_f32.keras", False,"float32")
train_and_save("fashion_mnist","models/fashion_model_f64.keras", epoch=25, batch_size=64)


