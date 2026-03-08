import numpy as np
import tensorflow as tf
import Utils
from tensorflow import keras
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


GOAL = 1e-15  #1e-6 float32-nél
IMG_INDEX = 0
NORM = False

clip = 255.0
lr = 10

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

original = x_test[IMG_INDEX]

if NORM:
    original = original / 255.0
    clip = 1.0

img = tf.Variable(original, dtype=tf.float64)
model = keras.models.load_model("models/mnist_model_f64.keras")
logit_model = Utils.build_logit_model(model)

originalOutput = logit_model(tf.expand_dims(img, 0), training=False)[0]

(_, c1), (_, c2) = Utils.two_largest(originalOutput)

while 1:
    with tf.GradientTape() as tape:
        tape.watch(img)

        preds = logit_model(tf.expand_dims(img, 0), training=False)[0]
        goal = tf.abs(preds[c1]-preds[c2])
        loss = tf.square(goal)

    if goal < GOAL:
        print(c1, ":", preds[c1])
        print(c2, ":", preds[c2])
        break

    grads = tape.gradient(loss, img)
    img.assign_sub(lr * grads)
    img.assign(tf.clip_by_value(img, 0.0, clip))

    print(
        "loss:", loss.numpy(),
        "grad max:", tf.reduce_max(tf.abs(grads)).numpy(),
        "goal:", goal,
    )

delta = img.numpy() - original

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

axes[0].imshow(original, cmap='gray')
axes[0].set_title("Eredeti")
axes[0].axis('off')

axes[1].imshow(img.numpy(), cmap='gray')
axes[1].set_title("Módosított")
axes[1].axis('off')

im = axes[2].imshow(delta, cmap='seismic')
axes[2].set_title("Pixelváltozás")
axes[2].axis('off')

cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.05)

plt.show()

img_name = "adv_image"+str(c1)+"_"+str(c2)
np.save(img_name, img.numpy())