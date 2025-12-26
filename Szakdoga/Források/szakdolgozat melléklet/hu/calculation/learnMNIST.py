import copy
import pickle
from tensorflow import keras
from tensorflow.keras import layers, models, initializers
import numpy as np
import tensorflow as tf

def lernAndGetWeight(model, name):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizálás [0,1] közé

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), shuffle=False)
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test

    errors = np.where(y_pred != y_true)[0]
    print(f"HOSSZA:{len(errors)}")
    with open("target/"+name+"_errors_detailed.txt", "w") as f:
        f.write("Index\tPredicted\tActual\n")
        for idx in errors:
            f.write(f"{idx}\t{y_pred[idx]}\t{y_true[idx]}\n")

    print(f"{len(errors)} hibás előrejelzés történt.")
    print("Részletek mentve.")


    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTeszt pontosság: {test_acc:.4f}")

    weights1 = []
    for layer in model.layers:
        weights = layer.get_weights()
        print(f"{layer.name} súlyai és biasai:")

        for w in weights:
            (weights1.append(w))

    writteWeights(weights1, name+".pkl")
    return weights1

def writeToFile(fileName, elteres):
    with open("target/"+fileName, 'w') as f:
        for item in elteres:
            for i in item:
                f.write(f"{i}\n")

def use_only_cpu():
    tf.config.set_visible_devices([], 'GPU')

def use_only_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    print("Elérhető fizikai GPU-k:", tf.config.list_physical_devices('GPU'))
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        return True
    return False

def reset_visible_devices():
    # visszaállít minden eszközt elérhetőre
    tf.config.set_visible_devices(tf.config.list_physical_devices())

def writteWeights(initial_weights, filename):
    with open("target/"+filename, "wb") as f:
        pickle.dump(initial_weights, f)

def readWeights(filename):
    # Súlyok visszaolvasása fájlból és visszaállítása
    with open("target/"+filename, "rb") as f:
        return pickle.load(f)

def basic_run(name, isFirstOne):
    model = models.Sequential([
            layers.Dense(128, activation="relu", kernel_initializer="he_uniform"),
            layers.Dense(64, activation="relu", kernel_initializer="he_uniform"),
            layers.Dense(10, activation="softmax", kernel_initializer="glorot_uniform")
        ])
    if isFirstOne:
        writteWeights(copy.deepcopy(model.get_weights()), "initial_weights.pkl")
    else:
        model.set_weights(readWeights("initial_weights.pkl"))


    w2 = lernAndGetWeight(model, name)
    if not isFirstOne:
        w1 = readWeights("original.pkl")
        elteres1 = []
        for i in range(len(w1)):
             elteres1.append( w1[i]- w2[i])
        writeToFile(name, elteres1)

def with_only_cpu():
    use_only_cpu()
    basic_run("only_cpu", False)

def with_only_gpu():
    use_only_gpu()
    basic_run("only_gpu", False)


# SEED = 42
# np.random.seed(SEED)
# random.seed(SEED)
# tf.random.set_seed(SEED)

basic_run("original", True)
#basic_run("second", False)

# Ha a device-t módosítani szeretnénk azokat kölön egyesével kell meghívni különben hibát dob
# Hiba: RuntimeError: Visible devices cannot be modified after being initialized
# Megoldás: a többi hívás legyen kikommentezve

# with_only_cpu()
# with_only_gpu()

