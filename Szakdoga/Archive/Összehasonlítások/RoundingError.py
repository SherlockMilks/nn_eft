import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import os

def set_seed(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

output_file = 'output.csv'

with open(output_file, 'w') as f:
    f.write("Original,Hand-calculated,Ascend,Descend\n")

    for i in range(1000):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5, verbose=0)

        #Tensorflow által számolt végeredmény
        final_output_original = model.predict(x_train[0:1])

        #Input kinyerése a manuális számításhoz
        input = x_train[0].reshape(-1)

        #Súlyok kinyerése a manuális számításhoz
        weights, bias = model.layers[1].get_weights()
        weights2, bias2 = model.layers[2].get_weights()
        weights3, bias3 = model.layers[3].get_weights()

        #Első réteg inputjának manipulálása, az összeadás sorrendjének változtatásával
        first_layer_input_normal = input @ weights + bias
        first_layer_input_ascend = np.array([
             np.sum(sorted(input * weights[:, i])) + bias[i] for i in range(weights.shape[1])
        ])
        first_layer_input_descend = np.array([
             np.sum(sorted(input * weights[:, i], reverse=True)) + bias[i] for i in range(weights.shape[1])
        ])

        #Első réteg manuálisan számolva
        first_layer_output_normal = np.maximum(0, first_layer_input_normal)
        first_layer_output_ascend = np.maximum(0, first_layer_input_ascend)
        first_layer_output_descend = np.maximum(0, first_layer_input_descend)

        #Második réteg manuálisan számolva
        second_layer_output_normal = np.maximum(0, first_layer_output_normal @ weights2 + bias2)
        second_layer_output_ascend = np.maximum(0, first_layer_output_ascend @ weights2 + bias2)
        second_layer_output_descend = np.maximum(0, first_layer_output_descend @ weights2 + bias2)

        #Utolsó réteg manuálisan számolva
        final_output_normal = tf.nn.softmax(second_layer_output_normal @ weights3 + bias3).numpy()
        final_output_ascend = tf.nn.softmax(second_layer_output_ascend @ weights3 + bias3).numpy()
        final_output_descend = tf.nn.softmax(second_layer_output_descend @ weights3 + bias3).numpy()

        #Eredmény kiírása a file-ba
        f.write(f"\"{','.join(map(str, final_output_original))}\","
                f"\"{','.join(map(str, final_output_normal))}\","
                f"\"{','.join(map(str, final_output_ascend))}\","
                f"\"{','.join(map(str, final_output_descend))}\"\n")
