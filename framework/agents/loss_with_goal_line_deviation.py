import tensorflow as tf
import numpy as np
from functools import partial
import typing

from problems import Problem
from .agent import GradientBasedAgent
from .util import get_distance_to_line


class LossWithGoalLineDeviation(GradientBasedAgent):
    """Implementation of a gradient-based agent with a custom loss function that tries to minimise the distance to the goal line.
    """

    def __init__(self, problem: Problem, learning_rate: float = 0.01, momentum: float = 0.0):
        """Constructor.

        Arguments:
            problem {Problem} -- the instance of the problem that this agent will train on

        Keyword Arguments:
            learning_rate {float} -- the learning rate (default: {0.01})
            momentum {float} -- the momentum (default: {0.0})
        """

        super().__init__(problem)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def compile(self):
        # Let us compile the keras model

        # We need to remember the initial prediction, so we can compute the goal line
        model = self.problem.model
        y_initial = model.predict(self.problem.X)

        # Define the custom loss function
        @tf.function
        def loss(y_true, y_pred):
            return tf.losses.mean_squared_error(y_true, y_pred) + get_distance_to_line(y_pred, y_initial, y_true) ** 2

        # Compile the model with the custom loss function
        model.compile(loss=loss,
                      optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate,
                                                        momentum=self.momentum),
                      metrics=["accuracy"])
