import typing
import numpy as np


class Metric:

    def __init__(self, name: str, dimensions: tuple = None):
        self.name = name
        self.dimensions = dimensions

    def __str__(self):
        if self.dimensions is None:
            return self.name
        return self.name + ":" + ":".join(map(str, self.dimensions))

    def select(self, metrics: typing.Dict[str, np.ndarray]) -> np.ndarray:
        metric = metrics[self.name]
        if self.dimensions is None:
            return metric
        else:
            return metric.__getitem__((..., *self.dimensions))

    @staticmethod
    def from_string(description: str):
        name, *dimensions = description.split(":")
        dimensions = tuple(map(int, dimensions))
        return Metric(name, dimensions if len(dimensions) > 0 else None)
