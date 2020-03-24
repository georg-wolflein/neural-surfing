from .problem import Problem
import numpy as np
import tensorflow as tf
from .util import DenseWithFixedBias


class ShallowProblem(Problem):

    def __init__(self):

        bias = 0.4
        initial_weights = np.array([.1, .0005])[..., np.newaxis]

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            DenseWithFixedBias(1, bias, initial_weights),
            tf.keras.layers.Activation("sigmoid")
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
        return np.squeeze(self.model.predict(self.X))
