from .problem import Problem
import numpy as np
import tensorflow as tf


class DenseWithFixedBias(tf.keras.layers.Layer):
    def __init__(self, num_outputs: int, bias: float, kernel: tf.Tensor):
        super().__init__()
        self.num_outputs = num_outputs
        self.bias_initializer = tf.constant_initializer(bias)
        self.kernel_initializer = tf.constant_initializer(kernel)

    def build(self, input_shape: tf.TensorShape):
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


class ShallowProblem(Problem):

    def __init__(self):

        bias = 0.4
        initial_weights = np.array([.1, .0005])[..., np.newaxis]

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            DenseWithFixedBias(1, bias, initial_weights)
        ])

        super().__init__(
            X=np.array([
                (.6, .4),
                (1., .4),
                (1., .4), (1., .4), (1., .4)
            ]),
            y=np.array([
                .2,
                .8,
                .8, .8, .8
            ]),
            model=model
        )

    @Problem.metric
    def weights(self):
        return tf.concat(
            [tf.reshape(x, [-1]) for x in self.model.trainable_weights],
            axis=0).numpy()

    @Problem.metric
    def output(self):
        return self.model.predict(self.X)
