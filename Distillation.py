import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha, temperature):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data

        teacher_pred = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)

            student_loss = self.student_loss_fn(y, student_pred)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_pred / self.temperature, axis=1),
                tf.nn.softmax(student_pred / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients,trainable_vars))

        self.compiled_metrics.update_state(y, student_pred)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distillation_lost":distillation_loss})

        return results

    def test_step(self, data):
        x, y = data

        y_pred = self.student(x, training=False)

        student_loss = self.student_loss_fn(y, y_pred)

        self.compiled_metrics.update_state(y, y_pred)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})

        return results


def makeStudent():
    student = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), dtype="float64"),
        tf.keras.layers.Dense(32, activation='relu', dtype="float64"),
        tf.keras.layers.Dense(16, activation='relu', dtype="float64"),
        tf.keras.layers.Dense(10, activation='softmax', dtype="float64")
    ])

    return student

def teachStudent(teacher, student):
    distiller = Distiller(student, teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10
    )

    distiller.fit(x_train, y_train, epochs=10)
    distiller.evaluate(x_test, y_test)

    student.save("mnist_model_student.keras")

teacher = keras.models.load_model("mnist_model_f64.keras")
teachStudent(teacher, makeStudent())

