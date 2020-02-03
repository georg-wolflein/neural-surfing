import matplotlib.pyplot as plt
import typing
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from .visualisation import Visualisation


class Scatter3D(Visualisation):

    def __init__(self, metric: str):
        super().__init__([metric], subplot_kwargs=dict(projection="3d"))
        self.metric = metric

    def setup(self, ax: plt.Axes, labels: typing.List[str], colors):
        ax.set_title(self.metric)
        ax.set_xlabel("Dimension #1")
        ax.set_ylabel("Dimension #2")
        ax.set_zlabel("Dimension #3")

        for label, color in zip(labels, colors):
            ax.scatter([], [], c=color, label=label)

        ax.legend()

    def plot(self, metrics: typing.Dict[str, np.ndarray], color, ax: plt.Axes):
        ax.scatter(*metrics[self.metric].T[:3], c=color)
