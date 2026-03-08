"""
Using the model and an image set, this script finds
the image for which the top two logits are closest in value.
"""

from tensorflow import keras
import matplotlib
import Utils
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# x_test = x_test/255.0

model = keras.models.load_model("models/mnist_model_f64.keras")
logit_model = Utils.build_logit_model(model)

min = float('inf')
min_idx = -1

for i in range(len(x_test)):

    output = logit_model.predict(x_test[i:i+1], verbose=0)
    output = output[0]

    (first, _), (second, _) = Utils.two_largest(output)

    diff = first - second

    if diff < min:
        min = diff
        min_idx = i


print(f"Smallest difference: {min}\n"
      f"The index of tha image: {min_idx}")


plt.figure(figsize=(6, 6))
plt.imshow(x_test[min_idx], cmap='gray')
plt.axis('off')
plt.show()
