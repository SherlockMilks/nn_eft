"""
This module implements different summation strategies (ascending,
descending, random order, random tree) and uses Knuth's two_sum
algorithm to track rounding errors during accumulation.
"""


import numpy as np
import random

np.set_printoptions(precision=64)

# Sorts the numbers in ascending order before summing them.
def ascend(values, *args):
    dtype = values[0].dtype

    values = values[values != 0]

    if len(values) == 0:
        return dtype.type(0), dtype.type(0)

    values_sorted = values[np.argsort(np.abs(values))]

    sum_ = np.sum(values_sorted, dtype=dtype)
    return sum_, dtype.type(0)

# Sorts the numbers in descending order before summing them.
def descend(values, *args):
    dtype = values[0].dtype

    values = values[values != 0]

    if len(values) == 0:
        return dtype.type(0), dtype.type(0)

    values_sorted = values[np.argsort(-np.abs(values))]

    sum_ = np.sum(values_sorted, dtype=dtype)
    return sum_, dtype.type(0)


# Computes a floating-point sum using randomized pairwise reduction
# with error tracking via the `two_sum` error-free transformation.
def random_pairwise_sum(values, k=1):
    dtype = values[0].dtype

    values = [value for value in values if value != 0]

    if not values:
        return dtype.type(0), dtype.type(0)

    errors = []

    while len(values) > 1:
        i, j = random.sample(range(len(values)), 2)

        i_num = values[i]
        j_num = values[j]

        sum_, e = two_sum(i_num, j_num)

        if e != 0:
            errors.append(e)

        values[i] = sum_

        del values[j]

    error_sums = []
    for i in range(0, k):
        if not errors:
            break
        error_sum = dtype.type(0)
        errors_temp = []
        for e in errors:
            error_sum, err = two_sum(error_sum, e)
            if err != 0:
                errors_temp.append(err)
        errors = errors_temp
        error_sums.append(error_sum)

    return values[0], error_sums


# Computes a floating-point sum using randomized sequential accumulation
# with error tracking via the `two_sum` error-free transformation.
def random_sequential_sum(values, k=1):
    dtype = values[0].dtype

    values = [value for value in values if value != 0]

    if not values:
        return dtype.type(0), dtype.type(0)

    errors = []
    sum_ = dtype.type(0)

    while len(values) > 0:
        i = random.randrange(len(values))

        i_num = values.pop(i)

        sum_, e = two_sum(sum_, i_num)

        if e != 0:
            errors.append(e)


    error_sums = []
    for i in range(0, k):
        if not errors:
            break
        error_sum = dtype.type(0)
        errors_temp = []
        for e in errors:
            error_sum, err = two_sum(error_sum, e)
            if err != 0:
                errors_temp.append(err)
        errors = errors_temp
        error_sums.append(error_sum)

    return sum_, error_sums


# Computes the output of a linear layer using a custom summation order,
# while tracking rounding errors for each neuron.
def linear_layer_custom_sum(input, weights, bias, sum_function, k=1):
    results = []
    all_errors = []

    for i in range(weights.shape[1]):
        result, errors = sum_function(input * weights[:, i], k)
        results.append(result + bias[i])
        all_errors.append(errors)

    return np.array(results), all_errors


# Knuths algorithm for the error-free transformation of
# the sum of two floating point numbers
def two_sum(a, b):
    x = a + b
    z = x - a
    e = (a - (x - z)) + (b - z)
    return x, e









