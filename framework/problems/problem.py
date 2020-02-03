from abc import ABC, abstractproperty, abstractmethod
import tensorflow as tf
import numpy as np
import typing


class Problem(ABC):

    def __init__(self, X: np.ndarray, y: np.ndarray, model: tf.keras.Model):
        self._X = X
        self._y = y
        self._model = model
        self._metrics = {}

        for x in dir(self):
            attr = getattr(self, x)
            if hasattr(attr, "_metric"):
                self._metrics[attr._metric] = attr

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    @staticmethod
    def metric(func: typing.Callable):
        func._metric = func.__name__
        return func

    def evaluate_metrics(self) -> typing.Dict[str, typing.Callable]:
        return {
            k: v() for (k, v) in self._metrics.items()
        }
