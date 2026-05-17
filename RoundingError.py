import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import DifferentOrders

np.set_printoptions(precision=64)

OUTPUT_FILE_BA = 'test1.csv'
OUTPUT_FILE_AA = 'test2.csv'
IMG_INDEX = 0
RND_AMOUNT = 1000
NORM = True
SIM_PARALLEL = False
K = 1  #K-Fold Value

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

sum_function = DifferentOrders.random_sequential_sum
if SIM_PARALLEL:
    sum_function = DifferentOrders.random_pairwise_sum

if NORM:
    x_test = x_test / 256.0

model = keras.models.load_model("models/mnist_model_norm2.h5")
input_dtype = model.layers[0].dtype

# Extracting the weights for manual calculations
weights = []
bias = []
for layer in model.layers:
    if isinstance(layer, Dense):
        w, b = layer.get_weights()

        if input_dtype == "float32":
            w = w.astype(np.float32)
            b = b.astype(np.float32)

        weights.append(w)
        bias.append(b)

def run_summation_orders(img):
    # TensorFlow result
    softmax_output_tf = model.predict(tf.expand_dims(img, 0), verbose=0)

    # Input extraction
    input_vec = img.reshape(-1).astype(input_dtype)

    logit_output_original = input_vec
    softmax_output_original = []
    logit_output_ascend = input_vec
    softmax_output_ascend = []
    logit_output_descend = input_vec
    softmax_output_descend = []

    for i in range(len(weights)):
        # Original, ascending and descending summation orders
        logit_output_original = logit_output_original @ weights[i] + bias[i]
        logit_output_ascend, _ = DifferentOrders.linear_layer_custom_sum(logit_output_ascend, weights[i], bias[i], DifferentOrders.ascend)
        logit_output_descend, _ = DifferentOrders.linear_layer_custom_sum(logit_output_descend, weights[i], bias[i], DifferentOrders.descend)

        if i == len(weights)-1:
            softmax_output_original = tf.nn.softmax(logit_output_original).numpy()
            softmax_output_ascend = tf.nn.softmax(logit_output_ascend).numpy()
            softmax_output_descend = tf.nn.softmax(logit_output_descend).numpy()
        else:
            logit_output_original = np.maximum(0, logit_output_original)
            logit_output_ascend = np.maximum(0, logit_output_ascend)
            logit_output_descend = np.maximum(0, logit_output_descend)


    # Calculating the results with randomized summation orders
    logit_outputs_random = []
    softmax_outputs_random = []
    logit_outputs_random_eft = []
    softmax_outputs_random_eft = []

    for i in range(RND_AMOUNT):
        logit_output_random = input_vec
        softmax_output_random = []
        logit_output_random_eft = input_vec
        softmax_output_random_eft = []

        for j in range(len(weights)):
            logit_output_random, _ = DifferentOrders.linear_layer_custom_sum(
                logit_output_random, weights[j], bias[j], sum_function)

            logit_output_random_eft, all_errors = DifferentOrders.linear_layer_custom_sum(
                logit_output_random_eft, weights[j], bias[j], sum_function, K)

            for i in range(len(all_errors)):
                error_sum = tf.constant(0, dtype=input_dtype)
                for e in reversed(all_errors[i]):
                    error_sum += e

                logit_output_random_eft[i] += error_sum

            if j == len(weights) - 1:
                softmax_output_random = tf.nn.softmax(logit_output_random).numpy()
                softmax_output_random_eft = tf.nn.softmax(logit_output_random_eft).numpy()

            else:
                logit_output_random = np.maximum(0, logit_output_random)
                logit_output_random_eft = np.maximum(0, logit_output_random_eft)


        logit_outputs_random.append(logit_output_random)
        softmax_outputs_random.append(softmax_output_random)
        logit_outputs_random_eft.append(logit_output_random_eft)
        softmax_outputs_random_eft.append(softmax_output_random_eft)


    # Writing the results into the output files
    with open(OUTPUT_FILE_BA, 'w') as f:
        f.write("Original\n")
        f.write(",".join(map(str, logit_output_original)) + "\n")

        f.write("Ascend\n")
        f.write(",".join(map(str, logit_output_ascend)) + "\n")

        f.write("Descend\n")
        f.write(",".join(map(str, logit_output_descend)) + "\n")

        for i in range(len(logit_outputs_random)):
            f.write(f"Random{i+1}\n")
            #f.write(",".join(map(str, random_outputs_raw[i])) + "\n")
            f.write("Raw:"+",".join(map(str, logit_outputs_random[i])) + "\n")
            f.write("EFT:" + ",".join(map(str, logit_outputs_random_eft[i])) + "\n")

    with open(OUTPUT_FILE_AA, 'w') as f:
        f.write("Tensorflow\n")
        f.write(",".join(map(str, softmax_output_tf[0])) + "\n")

        f.write("Original\n")
        f.write(",".join(map(str, softmax_output_original)) + "\n")

        f.write("Ascend\n")
        f.write(",".join(map(str, softmax_output_ascend)) + "\n")

        f.write("Descend\n")
        f.write(",".join(map(str, softmax_output_descend)) + "\n")

        for i in range(len(softmax_outputs_random)):
            f.write(f"Random{i+1}\n")
            #f.write(",".join(map(str, random_outputs_sm[i])) + "\n")
            f.write("Raw:" + ",".join(map(str, softmax_outputs_random[i])) + "\n")
            f.write("EFT:" + ",".join(map(str, softmax_outputs_random_eft[i])) + "\n")




# single_img = x_test[IMG_INDEX]
# run_summation_orders(single_img)

# adv_img = np.load("adversarial_img/fashion/adv_imagePullover_Coat.npy")
# run_summation_orders(adv_img)


for i in range(0,1000):
     OUTPUT_FILE_BA = 'output/eft/norm2/sequential/logit/modelnorm2_sequential_logit'
     OUTPUT_FILE_AA = 'output/eft/norm2/sequential/softmax/modelnorm2_sequential_softmax'

     idx = str(i)+".csv"
     OUTPUT_FILE_BA += idx
     OUTPUT_FILE_AA += idx
     run_summation_orders(x_test[i])






