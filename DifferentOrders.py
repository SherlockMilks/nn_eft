import numpy as np
import random

np.set_printoptions(precision=64)

def ascend(input, weights, bias, prints=False, file=None):
    if(prints):
        file.write(str(np.array([sorted((input * weights[:, i])[(input * weights[:, i]) != 0]) for i in range(weights.shape[1])]))+"\n")
    return np.array([np.sum(sorted((input * weights[:, i])[(input * weights[:, i]) != 0])) + bias[i] for i in range(weights.shape[1])])

def descend(input, weights, bias, prints=False, file=None):
    if(prints):
        file.write(str(np.array([sorted((input * weights[:, i])[(input * weights[:, i]) != 0], reverse=True) for i in range(weights.shape[1])]))+"\n")
    return np.array([np.sum(sorted((input * weights[:, i])[(input * weights[:, i]) != 0], reverse=True)) + bias[i] for i in range(weights.shape[1])])

def random_tree_sum(values, prints=False, file=None):
    values = [value for value in values if value != 0]

    if not values:
        return 0

    errors = []

    while len(values) > 1:
        i, j = random.sample(range(len(values)), 2)

        if i > j:
            i, j = j, i
        i_num = values[i]
        j_num = values[j]

        s, e = two_sum(i_num, j_num)

        if e != 0:
            errors.append(e)

        if prints and file is not None:
            file.write(f"{i_num}+{j_num}: {e}\n")

        values[i] = s

        del values[j]

    res = 0
    for i in errors:
        res, err = two_sum(res,i)
        if err != 0:
            print("Hiba:", err)

    return values[0], res

def randomOrder(input, weights, bias, prints=False, file=None):
    results = []
    all_errors = []

    for i in range(weights.shape[1]):
        result, errors = random_tree_sum(input * weights[:, i], prints, file)
        results.append(result + bias[i])
        all_errors.append(errors)

    return np.array(results), all_errors


def two_sum(a, b):
    x = a + b
    z = x - a
    e = (a - (x - z)) + (b - z)
    return x, e









