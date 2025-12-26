import numpy as np
import itertools as it


def signCalculate(weights, xVektor, sort):
    multiplied = weights * xVektor[:, np.newaxis].astype(np.float32)
    # Oszloponként rendezés
    sorted_matrix = multiplied
    if(sort):
        sorted_matrix = np.sort(multiplied, axis=0)
    # Oszloponkénti összegzés
    return np.sum(sorted_matrix, axis=0)



def relu(sign_vector):
    return np.where(sign_vector > 0, sign_vector, 0).astype(np.float32)

def relu_derivative(input_vector):
    return np.where(input_vector > 0, 1, 0).astype(np.float32)

def deviation(actual_vector, expected_vektor):
    return expected_vektor-actual_vector.astype(np.float32)

def deviation_link_back(sign_vector, deviation_vector, weights):
    relu_deriv = relu_derivative(sign_vector)
    hidden_delta = deviation_vector * relu_deriv
    result = weights @ hidden_delta
    return result.astype(np.float32)

def modify_weights(weights_before, deviation, vector_x, num):
    learning_rate = 0.01 / (1 + 0.1 * num)
    delta = learning_rate * np.outer(vector_x, deviation).astype(np.float32)
    w = weights_before + delta
    return w




def meanSquaredError(deviationVector):
    return np.sum(deviationVector ** 2).astype(np.float32)
def sumOptimalization(weights):
    if len(weights)>0:
        signMin = []
        signMax = []
        for i in range(len(weights[0])):
            wi = []
            print(f"A {i}. oszlopnak a szélső értékei")
            for j in range(len(weights)):
                wi.append(weights[j][i])
            s = getMinMax(wi,j)
            signMin.append(s[0])
            signMax.append(s[1])
        return [signMin, signMax]

def getMinMax(wi, j):
    permuatations = list(it.permutations(wi))
    max = float('-inf')
    strmin=""
    strmax=""
    min = float('inf')
    for p in permuatations:
        sum = 0
        for i in range(j+1):
            sum = sum+p[i]
        if sum > max:
            max = sum
            strmax = p
        if sum < min:
            min = sum
            strmin = p
    print(f"A minimum sorrendje: {strmin}")
    print(f"A értéke: {min}")
    print(f"A maximum sorrendje: {strmax}")
    print(f"A értéke: {max}")
    return [ min, max]

def getMinMaxSigns(weights, xVektor):
    xWithWeights = weights * xVektor[:, np.newaxis].astype(np.float32)
    print(f"a kalkulálás előtt: {xWithWeights}")
    return sumOptimalization(xWithWeights)





def removeBiasFromWeights(weights):
    return  weights[1:]

def layerCalculation(xVektor, startWeights):
    signs = getMinMaxSigns(startWeights, xVektor)
    # Különböző nagyságrendű súlyok
    yVektorMin = np.array(signs[0])
    print(yVektorMin)
    yVektorMax = np.array(signs[1])
    print(yVektorMax)

    print(f"eltérés: {yVektorMax - yVektorMin}")
    return [yVektorMin, yVektorMax]