"""Metrics for experiments.
"""

from __future__ import annotations
import typing
import numpy as np


class Metric:

    """Class representing a metric and meta-information about it.

    A metric is a tensor of arbitrary rank.
    However, when visualising a metric, we want a time series of scalar values.
    This means that we store some index into the dimensionality of the metric that will obtain a scalar value.
    For example, a metric of shape [5, 6, 7] could be indexed by [0, 0, 0] which would obtain the first element.
    """

    def __init__(self, name: str, dimensions: tuple = None):
        """Constructor.

        Arguments:
            name {str} -- the name of the metric

        Keyword Arguments:
            dimensions {tuple} -- the index into the dimensionality of the metric (see note above) (default: {None})
        """
        self.name = name
        self.dimensions = dimensions

    def __str__(self) -> str:
        """Obtain a string representation of the metric name.

        Returns:
            str -- the metric name
        """
        if self.dimensions is None:
            return self.name
        return self.name + ":" + ":".join(map(str, self.dimensions))

    def select(self, metrics: typing.Dict[str, np.ndarray]) -> np.ndarray:
        """Obtain the data for this metric using the index.

        Arguments:
            metrics {typing.Dict[str, np.ndarray]} -- a dictionary of metrics

        Returns:
            np.ndarray -- the metric
        """

        metric = metrics[self.name]
        if self.dimensions is None:
            return metric
        else:
            return metric.__getitem__((..., *self.dimensions))

    @staticmethod
    def from_string(description: str) -> Metric:
        """Create an instance of the Metric class from a string.

        The string will be of the form "name:a:b:c" where "name" is the name of the metric and a, b, c consitute the index.

        Arguments:
            description {str} -- the string representation of the metric

        Returns:
            Metric -- the Metric instance
        """

        name, *dimensions = description.split(":")
        dimensions = tuple(map(int, dimensions))
        return Metric(name, dimensions if len(dimensions) > 0 else None)
