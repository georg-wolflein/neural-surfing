import matplotlib.pyplot as plt
import typing
import numpy as np

from .visualisation import Visualisation


class Scatter3D(Visualisation):

    def __init__(self, metric: str):
        super().__init__([metric], subplot_kwargs=dict(projection="3d"))
        self.metric = metric

        # Import Axes3D which injects 3D support into matplotlib
        from mpl_toolkits.mplot3d import Axes3D

    def setup(self, ax: plt.Axes):
        ax.set_title(self.metric)
        ax.set_xlabel("Dimension #1")
        ax.set_ylabel("Dimension #2")
        ax.set_zlabel("Dimension #3")

    def plot(self, metrics: typing.Dict[str, np.ndarray], color, ax: plt.Axes):
        ax.scatter(*metrics[self.metric].T[:3], c=color)
