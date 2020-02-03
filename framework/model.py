import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import numpy as np
from multiprocessing import Queue
import typing


class Model:

    def __init__(self, model: tf.keras.models.Model):
        self.model = model

    def get_weights(self):
        return tf.concat(
            [tf.reshape(x, [-1]) for x in self.model.trainable_weights],
            axis=0)

    def train(self, x, *args, epochs: int, callbacks: typing.List[tf.keras.callbacks.Callback] = [], **kwargs):
        # Synchronously train

        weights = np.zeros(shape=(epochs, self.get_weights().shape[0]))
        y_pred = np.zeros(
            shape=(epochs, x.shape[0], *self.model.outputs[0].shape[1:]))

        def callback(epoch, logs={}):
            weights[epoch] = self.get_weights().numpy()
            # TODO: reuse computed outputs from keras
            # y_pred[epoch] = self.model.predict(x)

        history = self.model.fit(x, *args, **kwargs,
                                 epochs=epochs,
                                 callbacks=callbacks + [tf.keras.callbacks.LambdaCallback(on_epoch_end=callback)])

        return history, weights, y_pred


class DenseWithFixedBias(tf.keras.layers.Layer):
    def __init__(self, num_outputs, bias: float, kernel: tf.Tensor):
        super().__init__()
        self.num_outputs = num_outputs
        self.bias_initializer = tf.constant_initializer(bias)
        self.kernel_initializer = tf.constant_initializer(kernel)

    def build(self, input_shape):
        self.bias = self.add_weight("bias",
                                    shape=(),
                                    initializer=self.bias_initializer,
                                    trainable=False)
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs],
                                      initializer=self.kernel_initializer)

    def call(self, input):
        return tf.matmul(input, self.kernel) + self.bias


y = np.array([
    .2,
    .8,
    .8, .8, .8
])
x = np.array([
    (.6, .4),
    (1., .4),
    (1., .4), (1., .4), (1., .4)
])

bias = 0.45
initial_weights = np.array([.1, .0005])[..., np.newaxis]


model1 = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    DenseWithFixedBias(1, bias, initial_weights)
])


model1.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    DenseWithFixedBias(1, bias, initial_weights)
])
model2.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

model3 = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    DenseWithFixedBias(1, bias, initial_weights)
])


def get_subgoal(y_initial, y_true, progress: float):
    return y_initial + (y_true - y_initial) * progress


def get_distance_to_line(point, a, b):
    #print(point, a, b)
    pa = point - a
    ba = b - a
    t = tf.tensordot(pa, ba, axes=[[0, 1], [1, 0]]) / \
        tf.tensordot(ba, ba, axes=[[0, 1], [1, 0]])
    d = tf.norm(pa - t * ba)
    return d


def fractional_loss(y_initial):
    def loss(y_true, y_pred):
        # calculate goal line:
        #y_true = get_subgoal(y_initial, y_true, .3)
        return tf.keras.losses.mean_squared_error(y_true, y_pred) + get_distance_to_line(y_pred, y_initial, y_true) ** 2
    return loss


y_initial = model3.predict(x)
subgoal = get_subgoal(y_initial, tf.expand_dims(y, axis=-1), .3)
loss = fractional_loss(y_initial)

model3.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

models = [Model(model1), Model(model2), Model(model3)]

EPOCHS = 2
epoch = 0

N = 100000 // EPOCHS

COLORS = cm.rainbow(np.linspace(0, 1, len(models)))[:, np.newaxis, :]

for epoch_batch in range(N):
    for i, (m, c) in enumerate(zip(models, COLORS)):
        history, weights, y_pred = m.train(x, y,
                                           batch_size=len(y),
                                           epochs=EPOCHS,
                                           verbose=0)
        if epoch_batch == 0:
            plt.scatter(weights[:, 0], weights[:, 1], c=c, label=str(i))
            plt.legend()
        else:
            plt.scatter(weights[:, 0], weights[:, 1], c=c)
    plt.draw()
    plt.pause(0.001)
    epoch += EPOCHS
plt.show()
