from abc import ABC, abstractproperty, abstractmethod
import tensorflow as tf
import numpy as np
import typing


class Problem(ABC):
    """Abstract base class representing a neural problem.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, model: tf.keras.Model):
        """Constructor.

        Arguments:
            X {np.ndarray} -- the input matrix
            y {np.ndarray} -- the output targets
            model {tf.keras.Model} -- the keras model
        """

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
        """Get the input matrix.

        Returns:
            np.ndarray -- the input matrix
        """

        return self._X

    @property
    def y(self) -> np.ndarray:
        """Get the output targets.

        Returns:
            np.ndarray -- the output targets
        """

        return self._y

    @property
    def model(self) -> tf.keras.Model:
        """Get the keras model.

        Returns:
            np.ndarray -- the keras model
        """

        return self._model

    @staticmethod
    def metric(func: typing.Callable) -> typing.Callable:
        """Decorator function that should be used within a problem implementation to define metrics as functions.

        Arguments:
            func {typing.Callable} -- a function that calculates a metric

        Returns:
            typing.Callable -- the decorated function
        """

        func._metric = func.__name__
        return func

    def evaluate_metrics(self, metrics: typing.List[str] = None) -> typing.Dict[str, np.ndarray]:
        """Evaluate a subset of metrics.

        Keyword Arguments:
            metrics {typing.List[str]} -- the list of metrics to evaluate (None means all metrics) (default: {None})

        Returns:
            typing.Dict[str, np.ndarray] -- the evaluated metrics as a map from metric name to value(s)
        """
        return {
            k: v() for (k, v) in self._metrics.items()
            if metrics is None or k in metrics
        }
