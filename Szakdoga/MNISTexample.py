import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()


plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Szám: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()