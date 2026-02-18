import numpy as np
import tensorflow as tf
from tensorflow import keras
import DifferentOrders
from DifferentOrders import two_sum

np.set_printoptions(precision=64)

OUTPUT_FILE_BA = 'output/test1.csv'
OUTPUT_FILE_AA = 'output/test2.csv'
ADDITION_ERROR_FILE = "output/addition_error.csv"
PRINT_ADDITION_ERROR = True
IMG_INDEX = 0
RND_AMOUNT = 100
NORM = False
SIM_PARALLEL = False

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

sum_function = DifferentOrders.random_sum
if SIM_PARALLEL:
    sum_function = DifferentOrders.random_tree_sum

if NORM:
    x_test = x_test / 255.0

IMG = np.load("adversarial_img/f64/adv_image0_9.npy")
#IMG = x_test[IMG_INDEX]


model = keras.models.load_model("models/mnist_model_f64.keras")
input_dtype = model.layers[0].dtype

# Súlyok kinyerése a manuális számításhoz
weights, bias = model.layers[1].get_weights()
weights2, bias2 = model.layers[2].get_weights()
weights3, bias3 = model.layers[3].get_weights()

if input_dtype == "float32":
    weights = weights.astype(np.float32)
    bias = bias.astype(np.float32)
    weights2 = weights2.astype(np.float32)
    bias2 = bias2.astype(np.float32)
    weights3 = weights3.astype(np.float32)
    bias3 = bias3.astype(np.float32)


# TensorFlow által számolt végeredmény
final_output_original = model.predict(tf.expand_dims(IMG, 0), verbose=0)

# Input kinyerése a manuális számításhoz
input_vec = IMG.reshape(-1).astype(input_dtype)

# Eredeti sorrend
first_layer_normal = input_vec @ weights + bias
first_layer_normal = np.maximum(0, first_layer_normal)

second_layer_normal = first_layer_normal @ weights2 + bias2
second_layer_normal = np.maximum(0, second_layer_normal)

third_layer_normal = second_layer_normal @ weights3 + bias3
final_output_normal = tf.nn.softmax(third_layer_normal).numpy()


#Növekvő sorrend
first_layer_ascend = DifferentOrders.ascend(input_vec, weights, bias)
first_layer_ascend = np.maximum(0, first_layer_ascend)

second_layer_ascend = DifferentOrders.ascend(first_layer_ascend, weights2, bias2)
second_layer_ascend = np.maximum(0, second_layer_ascend)

third_layer_ascend = DifferentOrders.ascend(second_layer_ascend, weights3, bias3)
final_output_ascend = tf.nn.softmax(third_layer_ascend).numpy()


#Csökkenő sorrend
first_layer_descend = DifferentOrders.descend(input_vec, weights, bias)
first_layer_descend = np.maximum(0, first_layer_descend)

second_layer_descend = DifferentOrders.descend(first_layer_descend, weights2, bias2)
second_layer_descend = np.maximum(0, second_layer_descend)

third_layer_descend = DifferentOrders.descend(second_layer_descend, weights3, bias3)
final_output_descend = tf.nn.softmax(third_layer_descend).numpy()


with (open(ADDITION_ERROR_FILE, 'w') as g):
    #Randomizált összeadások kiszámítása
    random_outputs_sm = []
    random_outputs_raw = []
    random_outputs_eft = []
    for i in range(RND_AMOUNT):
        errors = []

        if PRINT_ADDITION_ERROR:
            g.write(f"Random{i + 1}\n")
            g.write(f"First layer\n")
        r, e = DifferentOrders.randomOrder(input_vec, weights, bias, sum_function, PRINT_ADDITION_ERROR, g)
        r = np.maximum(0, r)

        for i in range(len(r)):
            if r[i] == 0: e[i] = 0

        for i in range(weights2.shape[1]):
            errors.append([val for val in (e * weights2[:, i]) if val != 0])


        if PRINT_ADDITION_ERROR:
            g.write(f"Second layer\n")
        r, e = DifferentOrders.randomOrder(r, weights2, bias2, sum_function, PRINT_ADDITION_ERROR, g)
        r = np.maximum(0, r)

        for i in range(len(r)):
            if r[i] == 0: e[i] = 0

        for i in range(len(errors)):
            if e[i] != 0:
                errors[i].append(e[i])

        temp = errors
        errors = []
        for i in range(weights3.shape[1]):
            neuron_err = []
            for idx, list in enumerate(temp):
                neuron_err += [val * weights3[idx, i] for val in list]
            errors.append(neuron_err)


        if PRINT_ADDITION_ERROR:
            g.write(f"Third layer\n")
        r2, e = DifferentOrders.randomOrder(r, weights3, bias3, sum_function, PRINT_ADDITION_ERROR, g)
        r1 = tf.nn.softmax(r2).numpy()

        for i in range(len(errors)):
            if e[i] != 0:
                errors[i].append(e[i])


        error_sums = []
        for array in errors:
            s = 0
            for i in array:
                s, e = two_sum(s, i)
            error_sums.append(s)

        result = []
        for i in range(len(r2)):
            s, e = two_sum(r2[i], error_sums[i])
            result.append(s)

        random_outputs_sm.append(r1)
        random_outputs_raw.append(r2)
        random_outputs_eft.append(result)


    # Eredmények kiírása fileba
    with open(OUTPUT_FILE_BA, 'w') as f:
        f.write("Original\n")
        f.write(",".join(map(str, third_layer_normal)) + "\n")

        f.write("Ascend\n")
        f.write(",".join(map(str, third_layer_ascend)) + "\n")

        f.write("Descend\n")
        f.write(",".join(map(str, third_layer_descend)) + "\n")

        for i in range(len(random_outputs_raw)):
            f.write(f"Random{i+1}\n")
            #f.write(",".join(map(str, random_outputs_raw[i])) + "\n")
            f.write("Raw:"+",".join(map(str, random_outputs_raw[i])) + "\n")
            f.write("EFT:" + ",".join(map(str, random_outputs_eft[i])) + "\n")

    with open(OUTPUT_FILE_AA, 'w') as f:
        f.write("Tensorflow\n")
        f.write(",".join(map(str, final_output_original[0])) + "\n")

        f.write("Original\n")
        f.write(",".join(map(str, final_output_normal)) + "\n")

        f.write("Ascend\n")
        f.write(",".join(map(str, final_output_ascend)) + "\n")

        f.write("Descend\n")
        f.write(",".join(map(str, final_output_descend)) + "\n")

        for i in range(len(random_outputs_sm)):
            f.write(f"Random{i+1}\n")
            f.write(",".join(map(str, random_outputs_sm[i])) + "\n")
