import tensorflow as tf
import numpy as np
from functools import partial
import typing

from problems import Problem
from .agent import Agent


class MSE(Agent):

    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.compiled = False

    def compile(self, x):
        model = self.problem.model
        model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

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
