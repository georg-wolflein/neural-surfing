"""The classical steepest gradient descent agent with mean squared eror.
"""

import tensorflow as tf
import numpy as np
from functools import partial
import typing

from ..problems import Problem
from . import DerivativeBasedAgent


class MSE(DerivativeBasedAgent):
    """Implementation of the classical steepest gradient descent agent with mean squared eror.
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
        # We will simply compile using the SGD optimizer in keras
        self.problem.model.compile(loss="mse",
                                   optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate,
                                                                     momentum=self.momentum),
                                   metrics=["accuracy"])
