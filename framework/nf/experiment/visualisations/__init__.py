"""Visualisations for experiments.
"""

from abc import ABC, abstractmethod
import typing
import numpy as np
from bokeh.plotting import Figure
from bokeh.models.glyphs import Line
from bokeh.document import Document


class Visualisation(ABC):
    """Abstract base class for a visualisation.
    """

    def __init__(self, required_metrics: typing.List[str]):
        """Constructor.

        Arguments:
            required_metrics {typing.List[str]} -- the metrics required by the initialisation (this is how the visualisation registers for receiving specific data)
        """

        self.required_metrics = required_metrics

    @abstractmethod
    def setup(self) -> typing.Tuple[Figure, typing.List[Line]]:
        """An abstract method that will be called once to setup the visualisation with bokeh.

        Returns:
            typing.Tuple[Figure, typing.List[Line]] -- the plot and list of artists representing the agents on the plot
        """

    @abstractmethod
    def plot(self, metrics: typing.List[typing.Dict[str, np.ndarray]], plot: Figure, doc: Document):
        """Abstract method to update the plot with new data. 

        Arguments:
            metrics {typing.List[typing.Dict[str, np.ndarray]]} -- the relevant metrics
            plot {Figure} -- reference to the plot
            doc {Document} -- the enclosing bokeh document
        """
