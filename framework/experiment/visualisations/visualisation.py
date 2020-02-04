from abc import ABC, abstractmethod
import typing
import numpy as np
import matplotlib.pyplot as plt


class Visualisation(ABC):

    def __init__(self, required_metrics: typing.List[str], subplot_kwargs: dict = {}):
        self.required_metrics = required_metrics
        self.subplot_kwargs = subplot_kwargs

    @abstractmethod
    def setup(self, ax: plt.Axes):
        pass

    @abstractmethod
    def plot(self, metrics: typing.Dict[str, np.ndarray], color, ax: plt.Axes):
        pass
