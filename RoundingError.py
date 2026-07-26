import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import DifferentOrders
from pathlib import Path
import os
import argparse

np.set_printoptions(precision=64)

parser = argparse.ArgumentParser()
parser.add_argument("--dtype", type=str, default="bfloat16")
args = parser.parse_args()
input_dtype = np.dtype(args.dtype)

NORM = False
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
if NORM:
    x_test = x_test / 255.0

def run_summation_orders(img, model, input_dtype, output_file_ba='test1.csv', output_file_aa='test2.csv', rnd_amount=1000, sim_parallel=False, k=1):

    # Selecting sum function
    sum_function = DifferentOrders.random_sequential_sum
    if sim_parallel:
        sum_function = DifferentOrders.random_pairwise_sum

    # Extracting the weights for manual calculations
    weights = []
    bias = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            w, b = layer.get_weights()

            w = w.astype(input_dtype)
            b = b.astype(input_dtype)

            weights.append(w)
            bias.append(b)

    # Input extraction
    input_vec = img.reshape(-1).astype(input_dtype)

    logit_output_ascend = input_vec.copy()
    logit_output_ascend_eft = input_vec.copy()
    logit_output_descend = input_vec.copy()
    logit_output_descend_eft = input_vec.copy()

    for i in range(len(weights)):
        # Ascending and descending summation orders
        logit_output_ascend, _ = DifferentOrders.linear_layer_custom_sum(logit_output_ascend, weights[i], bias[i], DifferentOrders.ascend)

        logit_output_ascend_eft, ascend_error = DifferentOrders.linear_layer_custom_sum(logit_output_ascend_eft, weights[i], bias[i],
                                                                         DifferentOrders.ascend)
        logit_output_ascend_eft += ascend_error


        logit_output_descend, _ = DifferentOrders.linear_layer_custom_sum(logit_output_descend, weights[i], bias[i], DifferentOrders.descend)

        logit_output_descend_eft, descend_error = DifferentOrders.linear_layer_custom_sum(logit_output_descend_eft, weights[i], bias[i],
                                                                          DifferentOrders.descend)
        logit_output_descend_eft += descend_error

        if i != len(weights)-1:
            logit_output_ascend = np.maximum(0, logit_output_ascend)
            logit_output_ascend_eft = np.maximum(0, logit_output_ascend_eft)

            logit_output_descend = np.maximum(0, logit_output_descend)
            logit_output_descend_eft = np.maximum(0, logit_output_descend_eft)


    # Calculating the results with randomized summation orders
    logit_outputs_random = []
    logit_outputs_random_eft = []

    for i in range(rnd_amount):
        logit_output_random = input_vec.copy()
        logit_output_random_eft = input_vec.copy()

        for j in range(len(weights)):
            logit_output_random, _ = DifferentOrders.linear_layer_custom_sum(
                logit_output_random, weights[j], bias[j], sum_function)

            logit_output_random_eft, all_errors = DifferentOrders.linear_layer_custom_sum(
                logit_output_random_eft, weights[j], bias[j], sum_function, k)

            for idx in range(len(all_errors)):
                error_sum = input_dtype.type(0)
                for e in reversed(all_errors[idx]):
                    error_sum += e

                logit_output_random_eft[idx] += error_sum

            if j != len(weights) - 1:
                logit_output_random = np.maximum(0, logit_output_random)
                logit_output_random_eft = np.maximum(0, logit_output_random_eft)



        logit_outputs_random.append(logit_output_random)
        logit_outputs_random_eft.append(logit_output_random_eft)


    # Writing the results into the output files
    with open(output_file_ba, 'w') as f:
        f.write("Ascend\n")
        f.write("Raw:"+",".join(map(str, logit_output_ascend)) + "\n")
        f.write("EFT:"+",".join(map(str, logit_output_ascend_eft)) + "\n")

        f.write("Descend\n")
        f.write("Raw:"+",".join(map(str, logit_output_descend)) + "\n")
        f.write("EFT:"+",".join(map(str, logit_output_descend_eft)) + "\n")

        for i in range(len(logit_outputs_random)):
            f.write(f"Random{i+1}\n")
            f.write("Raw:"+",".join(map(str, logit_outputs_random[i])) + "\n")
            f.write("EFT:" + ",".join(map(str, logit_outputs_random_eft[i])) + "\n")


<<<<<<< HEAD

#Selecting models
dir = "models"
files = os.listdir(dir)
models = [file for file in files if file.endswith(".h5")]


=======
>>>>>>> bf2e5db836415ac8906dd9df593c84aa438d1ac2
# For multiple images from a dataset, run:
models_dir = "models"
files = os.listdir(models_dir)
models = [file for file in files if file.endswith(".h5")]

for model_name in models:
    model = keras.models.load_model(f"models/{model_name}")
    for i in range(0,1000):
        idx = str(i)+".csv"
        model_name_raw = model_name.replace(".h5","")
        output_dir_ba = Path(f'runs/eft/{input_dtype}/{model_name_raw}/sequential')
        output_dir_ba.mkdir(exist_ok=True, parents=True)

        output_file_ba = output_dir_ba / f"{model_name_raw+'_'+idx}"
        run_summation_orders(x_test[i], model, input_dtype, str(output_file_ba), rnd_amount=300)



# For one image from a dataset, run:
# worst_dir = "worst"
# files = os.listdir(worst_dir)
# images = [file for file in files if file.endswith(".png")]
#
# for img_name in images:
#     img_parts = img_name.split("_")
#     model_name_raw = img_parts[0]+"_"+img_parts[1]
#     model_name = model_name_raw+".keras"
#     img_idx = int(Path(img_parts[2]).stem)
#
#     single_img = x_test[img_idx]
#     model = keras.models.load_model(f"models/{model_name}")
#
#     output_dir_ba = Path(f'runs/worst/{input_dtype}/sequential')
#     output_dir_ba.mkdir(exist_ok=True, parents=True)
#
#     output_file_ba = output_dir_ba / f"{model_name_raw}.csv"
#     run_summation_orders(single_img, model, input_dtype, str(output_file_ba), sim_parallel=False, rnd_amount=1000)



# For an adversarial image, run:
# adv_dir = "adversarial_img"
# files = os.listdir(adv_dir)
# images = [file for file in files if file.endswith(".npy")]
#
#
# for img_name in images:
#     img_parts = img_name.split("_")
#     model_name_raw = img_parts[0]+"_"+img_parts[1]
#     model_name = model_name_raw+".keras"
#
#     adv_img = np.load(f"adversarial_img/{img_name}")
#     model = keras.models.load_model(f"models/{model_name}")
#
#     if int(Path(img_parts[2]).stem) == 0:
#         original = "base_first"
#     else:
#         original = "base_worst"
#
#     output_dir_ba = Path(f'runs/adversarial/{input_dtype}/sequential/{original}')
#     output_dir_ba.mkdir(exist_ok=True, parents=True)
#
#     output_file_ba = output_dir_ba / f"{model_name_raw}.csv"
#     run_summation_orders(adv_img, model, input_dtype, str(output_file_ba), sim_parallel=False, rnd_amount=1000)
#
#
#
#     output_dir_ba = Path(f'runs/adversarial/{input_dtype}/pairwise/{original}')
#     output_dir_ba.mkdir(exist_ok=True, parents=True)
#
#     output_file_ba = output_dir_ba / f"{model_name_raw}.csv"
#     run_summation_orders(adv_img, model, input_dtype, str(output_file_ba), sim_parallel=True, rnd_amount=1000)
