import numpy as np
from tensorflow import keras
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
img_index = 9752
original = x_test[img_index]

img = np.load("adversarial_img/mnist_D1_9752.npy")

delta = img - original
model = keras.models.load_model(f"models/mnist_D1.keras")

output = model.predict(np.expand_dims(img, axis=0))
print(output)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

axes[0].imshow(original, cmap='gray')
axes[0].set_title("Eredeti")
axes[0].axis('off')

axes[1].imshow(img, cmap='gray')
axes[1].set_title("Módosított")
axes[1].axis('off')

im = axes[2].imshow(delta, cmap='seismic')
axes[2].set_title("Pixelváltozás")
axes[2].axis('off')

cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.05)

plt.show()