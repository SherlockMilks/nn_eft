from copy import copy

import numpy as np


class NeuronNetworkModel:
    def __init__(self):
        self.weight_layer1 = np.random.normal(loc=0.0, scale=np.sqrt(2.0 / 128), size=(784, 128)).astype(np.float32)
        self.weight_layer2 = np.random.normal(loc=0.0, scale=np.sqrt(2.0 / 128),size=(128, 128)).astype(np.float32)
        self.weight_layer3 = np.random.normal(loc=0.0,scale=np.sqrt(2.0 / 128),size=(128, 10)).astype(np.float32)

    def get_weight_layer1(self):
        return copy(self.weight_layer1)
    def get_weight_layer2(self):
        return copy(self.weight_layer2)
    def get_weight_layer3(self):
        return copy(self.weight_layer3)

    def set_weight_layer1(self, weight):
        self.weight_layer1 = weight
    def set_weight_layer2(self, weight):
        self.weight_layer2 = weight
    def set_weight_layer3(self, weight):
        self.weight_layer3 = weight