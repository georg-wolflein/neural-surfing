from .problem import Problem
import numpy as np
import tensorflow as tf

# Radial basis activation function: e^(-x^2)
@tf.function
def rbf(x: tf.Tensor) -> tf.Tensor:
    return tf.exp(-tf.pow(x, 2))


class StripeProblem(Problem):

    def __init__(self):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1,
                                  use_bias=False,
                                  input_shape=(2,),
                                  activation=rbf)
        ])

        initial_weights = np.array([1, 1])[:, np.newaxis].astype(np.float32)
        model.weights[0].assign(tf.constant(initial_weights))

        super().__init__(
            X=np.array([
                (2, 2),
                (0, 2),
                (2, 0)
            ]).astype(np.float64),
            y=np.array([
                1,
                rbf(2.),
                rbf(2.)
            ]).astype(np.float64),
            model=model
        )

    @Problem.metric
    def weights(self):
        return tf.concat([tf.reshape(x, [-1])
                          for x in self.model.trainable_weights],
                         axis=0).numpy()

    @Problem.metric
    def output(self):
        return np.squeeze(self.model.predict(self.X))

    @Problem.metric
    def loss(self):
        return .25 * ((np.squeeze(self.model.predict(self.X)) - self.y) ** 2).sum()
