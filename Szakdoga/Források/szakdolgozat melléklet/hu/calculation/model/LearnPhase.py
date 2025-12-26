from copy import copy

import numpy as np

from hu.calculation.HyperGraph import sum_sub_sets, split_for_subset
from NeuronNetwork import relu, deviation_link_back
from NeuronNetwork import deviation
from NeuronNetwork import modify_weights
from NeuronNetworkModel import NeuronNetworkModel


class LearnPhase:
    def __init__(self, neuron_network_model:NeuronNetworkModel, input_vektor, expected_vektor, num):
        self.epoch_num = num
        self.weight= neuron_network_model
        self.sign_layer1=self.get_hyper_graph_sum(self.weight.get_weight_layer1(), input_vektor)
        self.y_layer1= relu(self.sign_layer1)
        self.sign_layer2=self.get_hyper_graph_sum(self.weight.get_weight_layer2(), self.y_layer1)
        self.y_layer2=relu(self.sign_layer2)
        self.sign_layer3=self.get_hyper_graph_sum(self.weight.get_weight_layer3(), self.y_layer2)
        self.y_layer3=relu(self.sign_layer3)
        self.learned_weight = NeuronNetworkModel()
        self.learn_weight(input_vektor,expected_vektor)

    def get_hyper_graph_sum(self,weights, x_vektor):
        signs = []
        for j in range(weights.shape[1]):
            weighted_column = weights[:, j] * x_vektor
            weighted_column = self.remove_zero(weighted_column)
            signs.append(sum_sub_sets(split_for_subset(list(weighted_column))))
        return np.array(signs)

    def remove_zero(self,lista):
        return [x for x in lista if x != 0]

    def learn_weight(self, x_vector, expected_vector):
        dev3 = deviation(self.y_layer3, expected_vector)
        new_w3 = modify_weights(self.weight.get_weight_layer3(), dev3, self.y_layer2, self.epoch_num)
        self.learned_weight.set_weight_layer3(new_w3)

        dev2 = deviation_link_back(self.sign_layer3, dev3, self.weight.get_weight_layer3())
        new_w2 = modify_weights(self.weight.get_weight_layer2(), dev2, self.y_layer1, self.epoch_num)
        self.learned_weight.set_weight_layer2(new_w2)

        dev1 = deviation_link_back(self.sign_layer2, dev2, self.weight.get_weight_layer2())
        new_w1 = modify_weights(self.weight.get_weight_layer1(), dev1, x_vector, self.epoch_num)
        self.learned_weight.set_weight_layer1(new_w1)



    def get_weight(self):
        return copy(self.weight)
    def get_sign_layer1(self):
        return copy(self.sign_layer1)
    def get_y_layer1(self):
        return copy(self.y_layer1)
    def get_sign_layer2(self):
        return copy(self.sign_layer2)
    def get_y_layer2(self):
        return copy(self.y_layer2)
    def get_learned_weight(self):
        return copy(self.learned_weight)

    def set_weight(self, w):
        self.weight = w
    def set_sign_layer1(self, w):
        self.sign_layer1 = w
    def set_y_layer1(self, w):
        self.y_layer1 = w
    def set_get_sign_layer2(self, w):
        self.sign_layer2 = w
    def get_sign_layer3(self):
        return self.sign_layer3
    def set_y_layer2(self, w):
        self.y_layer2 = w
    def set_learned_weight(self, w):
        self.learned_weight = w
