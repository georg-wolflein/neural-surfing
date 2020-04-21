from .problem import Problem
import numpy as np
import tensorflow as tf
from .util import DenseWithFixedBias


class SimpleProblem(Problem):
    """Implementation a simple neural problem that likely has no suboptimal local minima; can be used for testing that agents work under normal circumstances.
    """

    def __init__(self):

        # Initial weights and bias
        bias = 0
        initial_weights = np.array([0., 0.])[..., np.newaxis]

        # Define neural network architecture using keras
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            DenseWithFixedBias(1, bias, initial_weights)
        ])

        # Call problem constructor with the dataset
        super().__init__(
            X=np.array([
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1)
            ]).astype(np.float64),
            y=np.array([
                0,
                0.5,
                0.5,
                1
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
