import numpy as np
import tensorflow as tf
import Utils
from tensorflow import keras
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

model = keras.models.load_model("models/fashion_model_f64.keras")

print(model(tf.expand_dims(x_test[14], 0), training=False)[0])
print(y_test[14])