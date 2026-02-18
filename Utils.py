import tensorflow as tf
from tensorflow import keras

def build_logit_model(model):
    #Eredeti kimeneti layer
    last_dense = model.layers[-1]

    #Új, logitos kimeneti layer
    logit_layer = tf.keras.layers.Dense(
        units=last_dense.units,
        activation=None,
        use_bias=True,
        dtype=last_dense.dtype,
        name="logit_layer"
    )

    #Új layer rákötve a régiek végére
    logit_output = logit_layer(model.layers[-2].output)

    #Új student logit kimeneti layerrel
    logit_model = keras.Model(
        inputs=model.input,
        outputs=logit_output
    )
    logit_layer.set_weights(last_dense.get_weights())

    return logit_model


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



