"""
Utility functions used in other modules.
"""

import tensorflow as tf
from tensorflow import keras



# This function takes a model that applies an activation function
# to its output and returns a new model that outputs the logits instead.
def build_logit_model(model):
    # Original output layer
    last_dense = model.layers[-1]

    # New output layer that outputs logits
    logit_layer = tf.keras.layers.Dense(
        units=last_dense.units,
        activation=None,
        use_bias=True,
        dtype=last_dense.dtype,
        name="logit_layer"
    )

    # New layer is connected to previous layers
    logit_output = logit_layer(model.layers[-2].output)

    # The new model
    logit_model = keras.Model(
        inputs=model.input,
        outputs=logit_output
    )
    logit_layer.set_weights(last_dense.get_weights())

    return logit_model



# This function finds the two largest numbers in a list
# and returns the numbers with their indexes.
def two_largest(original):
    logit1_idx = 0
    logit2_idx = 0
    first = float('-inf')
    second = float('-inf')

    for i in range(len(original)):
        if original[i] > first:
            second = first
            logit2_idx = logit1_idx

            first = original[i]
            logit1_idx = i

        elif original[i] > second:
            second = original[i]
            logit2_idx = i


    return (first, logit1_idx), (second, logit2_idx)


def log_factory(file):
    def log(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=file)
    return log



