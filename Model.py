import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, x_val = x_train[:-5000], x_train[-5000:]
y_train, y_val = y_train[:-5000], y_train[-5000:]

def makeModel(norm=False,f32=False):
    global x_train
    global x_val
    global x_test

    if norm:
        if x_train.max() > 1:
            x_train = x_train / 255.0
            x_val = x_val / 255.0
            x_test = x_test / 255.0
    else:
        if x_train.max() < 1:
            x_train = x_train * 255.0
            x_val = x_val * 255.0
            x_test = x_test * 255.0

    dtype = "float32" if f32 else "float64"

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), dtype=dtype),
        tf.keras.layers.Dense(128, activation='relu', dtype=dtype),
        tf.keras.layers.Dense(64, activation='relu', dtype=dtype),
        tf.keras.layers.Dense(10, activation='softmax', dtype=dtype)
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_val, y_val))

    return model


def test_and_save(model, name):
    model.evaluate(x_test,y_test)
    model.save(name)


test_and_save(makeModel(),"mnist_model_f64.keras")
test_and_save(makeModel(True),"mnist_model_norm.keras")
test_and_save(makeModel(False,True),"mnist_model_f32.keras")


