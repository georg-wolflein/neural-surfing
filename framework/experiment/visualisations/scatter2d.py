import matplotlib.pyplot as plt
import typing
import numpy as np

from .visualisation import Visualisation


class Scatter2D(Visualisation):

    def __init__(self, metric: str):
        super().__init__([metric])
        self.metric = metric

    def setup(self, ax: plt.Axes, labels: typing.List[str], colors):
        ax.set_title(self.metric)
        ax.set_xlabel("Dimension #1")
        ax.set_ylabel("Dimension #2")

        for label, color in zip(labels, colors):
            ax.scatter([], [], c=color, label=label)

        ax.legend()

    def plot(self, metrics: typing.Dict[str, np.ndarray], color, ax: plt.Axes):
        ax.scatter(*metrics[self.metric].T[:2], c=color)
