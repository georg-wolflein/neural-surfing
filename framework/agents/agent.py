from abc import ABC, abstractmethod
import typing
import tensorflow as tf
import numpy as np

from problems import Problem


class Agent(ABC):

    def __init__(self, problem: Problem):
        self.problem = problem
        self.compiled = False

    @abstractmethod
    def compile(self):
        pass

    def train(self, epochs: int, metrics: typing.List[str] = None, start_epoch: int = 0):
        if not self.compiled:
            self.compile()
            self.compiled = True

        data = {}

        def callback(epoch, logs={}):
            calculated_metrics = self.problem.evaluate_metrics(metrics=metrics)
            if len(data) == 0:
                data.update({
                    name: np.zeros(shape=(epochs, *value.shape),
                                   dtype=value.dtype)
                    for name, value in calculated_metrics.items()
                })
            for name, value in calculated_metrics.items():
                data[name][epoch] = value

        self.problem.model.fit(self.problem.X, self.problem.y,
                               epochs=epochs,
                               callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=callback)])

        data["epoch"] = np.arange(start_epoch, start_epoch+epochs)

        return data
