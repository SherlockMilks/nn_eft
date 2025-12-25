import numpy as np
import tensorflow as tf
from tensorflow import keras
import DifferentOrders
from DifferentOrders import two_sum
from decimal import Decimal, getcontext


getcontext().prec = 20
np.set_printoptions(precision=64)

OUTPUT_FILE_BA = 'output/a_modelf64kk.csv'
OUTPUT_FILE_AA = 'output/a_modelf64.csv'
ADDITION_ERROR_FILE = "output/addition_error.csv"
PRINT_ADDITION_ERROR = True
IMG_INDEX = 0
RND_AMOUNT = 3
NORM = False



(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

if NORM:
    x_test = x_test / 255.0

model = keras.models.load_model("mnist_model_f64.keras")
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

with (open(ADDITION_ERROR_FILE, 'w') as g):
    # TensorFlow által számolt végeredmény
    final_output_original = model.predict(x_test[IMG_INDEX:IMG_INDEX + 1], verbose=0)

    # Input kinyerése a manuális számításhoz
    input_vec = x_test[IMG_INDEX].reshape(-1).astype(input_dtype)

    # Eredeti sorrend
    if not PRINT_ADDITION_ERROR:
        g.write("Original order\n")
        g.write(f"First layer\n")
        for i in range(weights.shape[1]):
            order = input_vec * weights[:, i]
            g.write(",".join(str(x) for x in order[order != 0]) + "\n")
    first_layer_normal = input_vec @ weights + bias
    first_layer_normal = np.maximum(0, first_layer_normal)

    if not PRINT_ADDITION_ERROR:
        g.write(f"Second layer\n")
        for i in range(weights2.shape[1]):
            order = first_layer_normal * weights2[:, i]
            g.write(",".join(str(x) for x in order[order != 0]) + "\n")
    second_layer_normal = first_layer_normal @ weights2 + bias2
    second_layer_normal = np.maximum(0, second_layer_normal)

    if not PRINT_ADDITION_ERROR:
        g.write(f"Third layer\n")
        for i in range(weights3.shape[1]):
            order = second_layer_normal * weights3[:, i]
            g.write(",".join(str(x) for x in order[order != 0]) + "\n")
    third_layer_normal = second_layer_normal @ weights3 + bias3
    final_output_normal = tf.nn.softmax(third_layer_normal).numpy()


    #Növekvő sorrend
    if not PRINT_ADDITION_ERROR:
        g.write("Ascend\n")
        g.write(f"First layer\n")
    first_layer_ascend = DifferentOrders.ascend(input_vec, weights, bias, not PRINT_ADDITION_ERROR, g)
    first_layer_ascend = np.maximum(0, first_layer_ascend)

    if not PRINT_ADDITION_ERROR:
        g.write(f"Second layer\n")
    second_layer_ascend = DifferentOrders.ascend(first_layer_ascend, weights2, bias2, not PRINT_ADDITION_ERROR, g)
    second_layer_ascend = np.maximum(0, second_layer_ascend)

    if not PRINT_ADDITION_ERROR:
        g.write(f"Third layer\n")
    third_layer_ascend = DifferentOrders.ascend(second_layer_ascend, weights3, bias3, not PRINT_ADDITION_ERROR, g)
    final_output_ascend = tf.nn.softmax(third_layer_ascend).numpy()


    #Csökkenő sorrend
    if not PRINT_ADDITION_ERROR:
        g.write("Descend\n")
        g.write(f"First layer\n")
    first_layer_descend = DifferentOrders.descend(input_vec, weights, bias, not PRINT_ADDITION_ERROR, g)
    first_layer_descend = np.maximum(0, first_layer_descend)

    if not PRINT_ADDITION_ERROR:
        g.write(f"Second layer\n")
    second_layer_descend = DifferentOrders.descend(first_layer_descend, weights2, bias2, not PRINT_ADDITION_ERROR, g)
    second_layer_descend = np.maximum(0, second_layer_descend)

    if not PRINT_ADDITION_ERROR:
        g.write(f"Third layer\n")
    third_layer_descend = DifferentOrders.descend(second_layer_descend, weights3, bias3, not PRINT_ADDITION_ERROR, g)
    final_output_descend = tf.nn.softmax(third_layer_descend).numpy()


    #Randomizált összeadások kiszámítása
    random_outputs_sm = []
    random_outputs = []
    for i in range(RND_AMOUNT):
        errors = []

        if PRINT_ADDITION_ERROR:
            g.write(f"Random{i + 1}\n")
            g.write(f"First layer\n")
        r, e = DifferentOrders.randomOrder(input_vec, weights, bias, PRINT_ADDITION_ERROR, g)
        r = np.maximum(0, r)
        for i in range(len(r)):
            if r[i] == 0: e[i] = 0

        for i in range(weights2.shape[1]):
            errors.append([val for val in (e * weights2[:, i]) if val != 0])


        if PRINT_ADDITION_ERROR:
            g.write(f"Second layer\n")
        r, e = DifferentOrders.randomOrder(r, weights2, bias2, PRINT_ADDITION_ERROR, g)
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
        r2, e = DifferentOrders.randomOrder(r, weights3, bias3, PRINT_ADDITION_ERROR, g)
        r1 = tf.nn.softmax(r2).numpy()

        for i in range(len(errors)):
            if e[i] != 0:
                errors[i].append(e[i])


        #csak 2 körös helyett ezt olyanra ami addig megy amíg epszilon értéknél nem kisebb az error
        error_sums = []
        error_sums2 = []
        for array in errors:
            sum=0
            new_errors=[]
            for i in array:
                sum, e = two_sum(sum,i)
                if e != 0:
                    new_errors.append(e)
            error_sums.append(sum)

            sum=0
            for i in new_errors:
                sum, e = two_sum(sum,i)
                if e != 0:
                    print("Hiba")
            error_sums2.append(sum)

        result=[]
        for i in range(len(r2)):
            decimal_r2 = Decimal(r2[i])
            decimal_e1 = Decimal(error_sums[i])
            decimal_e2 = Decimal(error_sums2[i])
            result.append(decimal_r2+decimal_e1+decimal_e2)
            print(result[i])

        random_outputs_sm.append(r1)
        random_outputs.append(result)


    # Eredmények kiírása fileba
    with open(OUTPUT_FILE_BA, 'w') as f:
        f.write("Original\n")
        f.write(",".join(map(str, third_layer_normal)) + "\n")

        f.write("Ascend\n")
        f.write(",".join(map(str, third_layer_ascend)) + "\n")

        f.write("Descend\n")
        f.write(",".join(map(str, third_layer_descend)) + "\n")

        for i, r_out in enumerate(random_outputs, 1):
            f.write(f"Random{i}\n")
            f.write(",".join(map(str, r_out)) + "\n")

    with open(OUTPUT_FILE_AA, 'w') as f:
        f.write("Tensorflow\n")
        f.write(",".join(map(str, final_output_original[0])) + "\n")

        f.write("Original\n")
        f.write(",".join(map(str, final_output_normal)) + "\n")

        f.write("Ascend\n")
        f.write(",".join(map(str, final_output_ascend)) + "\n")

        f.write("Descend\n")
        f.write(",".join(map(str, final_output_descend)) + "\n")

        for i, r_out in enumerate(random_outputs_sm, 1):
            f.write(f"Random{i}\n")
            f.write(",".join(map(str, r_out)) + "\n")
