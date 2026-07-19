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


def build_model(hidden_layers, width, dtype="float32"):

    layers = [
        tf.keras.layers.Flatten(input_shape=(28,28), dtype=dtype)
    ]

    for _ in range(hidden_layers):
        layers.append(
            tf.keras.layers.Dense(
                width,
                activation="relu",
                dtype=dtype
            )
        )

    layers.append(
        tf.keras.layers.Dense(
            10,
            activation="softmax",
            dtype=dtype
        )
    )

    model = tf.keras.Sequential(layers)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_and_save(dataset, name, hidden_layers=2, width=64, norm=False, dtype="float32", epoch=10, batch_size=32):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prep_data(dataset, norm)

    model = build_model(hidden_layers, width, dtype)

    model.fit(x_train, y_train, epochs=epoch, verbose=1, batch_size=batch_size, validation_data=(x_val, y_val))

    model.evaluate(x_test, y_test)
    model.save(name)



train_and_save("mnist","models/mnist_W16.keras", 2, 16)
train_and_save("mnist","models/mnist_W32.keras", 2, 32)
train_and_save("mnist","models/mnist_W64.keras", 2, 64)
train_and_save("mnist","models/mnist_W128.keras", 2, 128)
train_and_save("mnist","models/mnist_W256.keras", 2, 256)


train_and_save("mnist","models/mnist_D1.keras", 1, 32)
train_and_save("mnist","models/mnist_D2.keras", 2, 32)
train_and_save("mnist","models/mnist_D3.keras", 3, 32)
train_and_save("mnist","models/mnist_D4.keras", 4, 32)
train_and_save("mnist","models/mnist_D5.keras", 5, 32)



