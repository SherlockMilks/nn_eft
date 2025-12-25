import numpy as np
from tensorflow import keras
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# x_test = x_test/255.0

model = keras.models.load_model("mnist_model_f64.keras")

min = float('inf')
min_idx = -1

for i in range(len(x_test)):
    output = model.predict(x_test[i:i+1], verbose=0)
    output = output[0]

    sorted = np.sort(output)
    first = sorted[len(output)-1]
    second = sorted[len(output)-2]

    diff = first - second

    if diff < min:
        min = diff
        min_idx = i


print(f"Legkisebb különbség: {min}\n"
      f"Kép indexe: {min_idx}")


plt.figure(figsize=(6, 6))
plt.imshow(x_test[min_idx], cmap='gray')
plt.axis('off')
plt.show()
