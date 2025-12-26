import random
import numpy as np

def get_random_elem_and_remove(lista):
    random_index = random.randrange(len(lista))
    item = lista.pop(random_index)
    return item

def get_subset(lista):
    element_num = round(len(lista) * random.random())
    element_num = element_num if element_num != 0 else len(lista)
    subset = []
    for _ in range(element_num):
        subset.append(get_random_elem_and_remove(lista))
    return subset

def split_for_subset(lista):
    subsets = []
    i = 0
    while len(lista)!=0:
        i = i+1
        subset = get_subset(lista)
        if len(subset)>2:
            subset = split_for_subset(subset)
        subsets.append(subset)
    return subsets

def sum_sub_sets(subsets):
    amount = np.float32(0.0)
    for subset in subsets:
        if isinstance(subset, list):
            n = sum_sub_sets(subset)
            amount = amount + np.float32(n)
        else:
            amount = amount + np.float32(subset)
    return amount

