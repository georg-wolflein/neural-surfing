import tensorflow as tf
import numpy as np
from functools import partial
import typing

from problems import Problem
from .agent import Agent


def get_subgoal(y_initial, y_true, progress: float):
    return y_initial + (y_true - y_initial) * progress


def get_distance_to_line(point, a, b):
    pa = point - a
    ba = b - a
    t = tf.tensordot(pa, ba, axes=[[0, 1], [1, 0]]) / \
        tf.tensordot(ba, ba, axes=[[0, 1], [1, 0]])
    d = tf.norm(pa - t * ba)
    return d


class LossWithGoalLineDeviation(Agent):

    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.compiled = False

    def compile(self, x):
        model = self.problem.model
        y_initial = model.predict(x)

        def loss(y_true, y_pred):
            return tf.losses.mean_squared_error(y_true, y_pred) + get_distance_to_line(y_pred, y_initial, y_true) ** 2

        model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

        self.compiled = True

    def train(self, x, *args, epochs: int, callbacks: typing.List[tf.keras.callbacks.Callback] = [], **kwargs):
        if not self.compiled:
            self.compile(x)

        data = {}

        def callback(epoch, logs={}):
            metrics = self.problem.evaluate_metrics()
            if len(data) == 0:
                data.update({
                    name: np.zeros(shape=(epochs, *value.shape),
                                   dtype=value.dtype)
                    for name, value in metrics.items()
                })
            for name, value in metrics.items():
                data[name][epoch] = value

        history = self.problem.model.fit(x, *args, **kwargs,
                                         epochs=epochs,
                                         callbacks=callbacks + [tf.keras.callbacks.LambdaCallback(on_epoch_end=callback)])

        return history, data
