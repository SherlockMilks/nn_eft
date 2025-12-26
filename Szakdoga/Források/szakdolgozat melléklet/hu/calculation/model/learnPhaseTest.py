from copy import copy

from matplotlib import pyplot as plt

from hu.calculation.histogram import histogramDraw
from hu.calculation.model.LearnPhase import LearnPhase
from hu.calculation.model.NeuronNetworkModel import NeuronNetworkModel
import tensorflow as tf

def get_input_vector(image):
    input_vector = image.astype("float32") / 255.0  # értékek 0.0–1.0
    return input_vector.reshape(-1)
def proccess_data(data):
    lista = []
    for d in data:
        for i in d:
            lista.append(i)
    return list(lista)

model = NeuronNetworkModel()
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()


learn_phases_1 : [LearnPhase]= []
learn_phases_2 : [LearnPhase]= []
size = 5

for i in range(size):
    print(f"Iteráció: {i}")
    if i == 0:
        learn_phases_1.append(LearnPhase(copy(model), get_input_vector(x_train[i]), y_train[i], i+1))
        learn_phases_2.append(LearnPhase(copy(model), get_input_vector(x_train[i]), y_train[i], i+1))
    else:
        learn_phases_1.append(LearnPhase(copy(learn_phases_1[i-1].get_learned_weight()) , get_input_vector(x_train[i]), y_train[i], i+1))
        learn_phases_2.append(LearnPhase(copy(learn_phases_2[i-1].get_learned_weight()) , get_input_vector(x_train[i]), y_train[i], i+1))
####################################### EZ ITT A JELEK ÖSSZEVETÉSEÉHEZ VAN ##################################################################
for i in range(size):
    if i == 0 or i == 29:
        data_1 = learn_phases_1[i].get_sign_layer1() - learn_phases_2[i].get_sign_layer1()
        data_2 = learn_phases_1[i].get_sign_layer2() - learn_phases_2[i].get_sign_layer2()
        data_3 = learn_phases_1[i].get_sign_layer3() - learn_phases_2[i].get_sign_layer3()
        data_combined = data_1.tolist() + data_2.tolist() + data_3.tolist()
        header = "Eltérés a nerunok kimenetében:"
        histogramDraw(data_combined,header + str(i+1) + ". iteráció", 15)
        histogramDraw(data_1.tolist(), header + " az első rétegen",15)
        histogramDraw(data_2.tolist(), header + " a második rétegen",15)
        # histogramDraw(data_3.tolist(), header + " a harmadik rétegen",15)

####################################### EZ ITT A SÚLYOK ÖSSZEVETÉSÉHEZ VAN ################################################################
for i in range(size):
    if i == 0 or i == 29 or i == 4 or i==49:
        data_1 = learn_phases_1[i].get_learned_weight().get_weight_layer1()-learn_phases_2[i].get_learned_weight().get_weight_layer1()
        data_2 = learn_phases_1[i].get_learned_weight().get_weight_layer2()-learn_phases_2[i].get_learned_weight().get_weight_layer2()
        data_3 = learn_phases_1[i].get_learned_weight().get_weight_layer3()-learn_phases_2[i].get_learned_weight().get_weight_layer3()
        header = str(i+1)+ ". iteráció - Eltérés "
        # histogramDraw(proccess_data(data_1), header + "az első rétegen",15)
        # histogramDraw(proccess_data(data_2), header + "a második rétegen",15)
        # histogramDraw(proccess_data(data_3), header + "a harmadik rétegen",15)
        plt.figure(figsize=(12, 3))

        plt.subplot(1, 3, 1)
        plt.hist(proccess_data(data_1), bins=30, alpha=0.7, color='#0a6ba0', edgecolor='black')
        plt.title(header + "az első rétegen")

        plt.subplot(1, 3, 2)
        plt.hist(proccess_data(data_2), bins=30, alpha=0.7, color='#0a6ba0', edgecolor='black')
        plt.title(header + "a második rétegen")

        plt.subplot(1, 3, 3)
        plt.hist(proccess_data(data_3), bins=30, alpha=0.7, color='#0a6ba0', edgecolor='black')
        plt.title(header + "a harmadik rétegen")

        plt.tight_layout()
        # plt.show()
        plt.savefig("target\\" + header + ".png", dpi=300, bbox_inches='tight', transparent=False)