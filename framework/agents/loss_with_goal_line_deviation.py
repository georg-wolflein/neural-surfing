import tensorflow as tf
import numpy as np
from functools import partial
import typing

from problems import Problem
from .agent import GradientBasedAgent


def get_subgoal(y_initial, y_true, progress: float):
    return y_initial + (y_true - y_initial) * progress


def get_distance_to_line(point, a, b):
    pa = point - a
    ba = b - a
    t = tf.tensordot(pa, ba, axes=[[0, 1], [1, 0]]) / \
        tf.tensordot(ba, ba, axes=[[0, 1], [1, 0]])
    d = tf.norm(pa - t * ba)
    return d


class LossWithGoalLineDeviation(GradientBasedAgent):

    def compile(self):
        model = self.problem.model
        y_initial = model.predict(self.problem.X)

        def loss(y_true, y_pred):
            return tf.losses.mean_squared_error(y_true, y_pred) + get_distance_to_line(y_pred, y_initial, y_true) ** 2

        model.compile(loss=loss, optimizer="sgd", metrics=["accuracy"])
