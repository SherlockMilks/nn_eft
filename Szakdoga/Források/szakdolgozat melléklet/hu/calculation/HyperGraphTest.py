import HyperGraph as hg
import random
import numpy as np
import tensorflow as tf
import histogram as h

#################################################################################################################
### Azt vizsgálom hogy ha a sum 0-hoz nagyon közeli szám akkor a relatív hiba nagyon nagy #######################
## ebben az esetben hasonló nagyságrendű számokat adok össze, ugyanannyi pozitív és ugyan annyi negatív számot ##
#################################################################################################################
def sum_to_Zero():
    original=[]
    for i in range(128):
        original.append(1)
        original.append((-1+random.uniform(-1e-10, 1e-10)))
    calculation(original)

#################################################################################################################
### Azt vizsgálom hogy különböző nagyságrendű számok hogyan vislkednek ##########################################
#################################################################################################################

def different_magnitudes_num():
    original=[]
    for i in range(64):
        original.append(random.uniform(-1e-10, 1e-10))
        original.append(random.uniform(-1e+10, 1e+10))
    calculation(original)

#################################################################################################################
def split_almost_equal(lst, num_chunks):
    avg = len(lst) // num_chunks
    remainder = len(lst) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + avg + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks

def basic_he_inic():
    fan_in = 128  # bemeneti kapcsolatok száma
    he_std = np.sqrt(2.0 / fan_in)
    original = np.random.normal(loc=0.0, scale=he_std, size=128)
    calculation(original)

def mnist():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    # Első kép kiválasztása
    image = x_train[150]  # shape: (28, 28)

    # Normalizálás és lapítása
    input_vector = image.astype("float32") / 255.0  # értékek 0.0–1.0
    input_vector = input_vector.reshape(-1)  # shape: (784,)

    input_dim = 784  # MNIST bemeneti dimenzió
    limit = np.sqrt(6 / input_dim)

    # Egyetlen súlyvektor inicializálása (784 hosszú)
    weights = np.random.normal(loc=0.0, scale=np.sqrt(2.0 / 128), size=784)
    original = []
    for i in range(len(input_vector)):
        # print(f"A {type(input_vector[i])}")
        original.append(input_vector[i]*weights[i])

    calculation(remove_zero(original))


def random_from_weights():
    weights = np.random.normal(loc=0.0, scale=np.sqrt(2.0 / 128), size=784)
    original = []
    num_one = []
    for _ in range(158):
        original.append(hg.get_random_elem_and_remove(list(weights)))
        num_one.append(1)
    calculation(original)
    result = []
    for i in range(len(original)):
        result.append(original[i]*1)
    calculation(result)



def remove_zero(lista):
    return  [x for x in lista if x != 0]

def calculation(ORIGINAL):
    print(ORIGINAL)
    print(f"az abszolut érték átlaga: {sum(abs(x) for x in ORIGINAL) / len(ORIGINAL)}")
    print(f"az abszolut érték összege: {sum(abs(x) for x in ORIGINAL)}")
    print(f"az átlaga: {sum(ORIGINAL)/ len(ORIGINAL)}")
    print(f"a hossza: {len(ORIGINAL)}")
    result1 = hg.sum_sub_sets(hg.split_for_subset(list(ORIGINAL)))
    result2 = hg.sum_sub_sets(hg.split_for_subset(list(ORIGINAL)))
    result3 = hg.sum_sub_sets(hg.split_for_subset(list(ORIGINAL)))

    print(f"result 1: {result1}")
    print(f"result 2: {result2}")
    print(f"result 3: {result3}")
    print(f"eltérés 1-2: {result1 - result2}")
    print(f"eltérés 1-3: {result1 - result3}")
    print(f"eltérés 2-3: {result2 - result3}")


mnist()