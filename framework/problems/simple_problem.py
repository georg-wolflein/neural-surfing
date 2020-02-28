from .problem import Problem
import numpy as np
import tensorflow as tf


class SimpleProblem(Problem):

    def __init__(self):

        # Make sure the random seeds are the same for each run
        np.random.seed(1)
        tf.random.set_seed(2)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1, input_shape=(2,))
        ])

        super().__init__(
            X=np.array([
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1)
            ]).astype(np.float64),
            y=np.array([
                1,
                1,
                0,
                0
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
