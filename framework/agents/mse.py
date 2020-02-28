import tensorflow as tf
import numpy as np
from functools import partial
import typing

from problems import Problem
from .agent import Agent


class MSE(Agent):

    def compile(self):
        self.problem.model.compile(
            loss="mse", optimizer="sgd", metrics=["accuracy"])
