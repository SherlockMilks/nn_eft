import tensorflow as tf
from tensorflow import keras

#tf.debugging.set_log_device_placement(True)

output_file = "parallel_output.csv"

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32")

model = keras.models.load_model("mnist_model.keras")

img = tf.convert_to_tensor(x_test[0:1])

with open(output_file, 'w') as f:
    f.write("MNIST first image results\n")
    for i in range(10000):
        final_output = model.predict(img)
        concrete = tf.function(model).get_concrete_function(tf.TensorSpec(img.shape, img.dtype))
        f.write(f"{','.join(map(str, final_output[0]))}\n")
        for op in concrete.graph.get_operations():
            print(op.name, op.type)
