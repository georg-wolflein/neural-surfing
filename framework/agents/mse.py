import tensorflow as tf
import numpy as np
from functools import partial
import typing

from problems import Problem
from .agent import GradientBasedAgent


class MSE(GradientBasedAgent):

    def __init__(self, problem: Problem, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(problem)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def compile(self):
        self.problem.model.compile(loss="mse",
                                   optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate,
                                                                     momentum=self.momentum),
                                   metrics=["accuracy"])
