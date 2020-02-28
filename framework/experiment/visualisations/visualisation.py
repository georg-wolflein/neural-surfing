from abc import ABC, abstractmethod
import typing
import numpy as np
from bokeh.plotting import Figure
from bokeh.models.glyphs import Line
from bokeh.document import Document


class Visualisation(ABC):

    def __init__(self, required_metrics: typing.List[str]):
        self.required_metrics = required_metrics

    @abstractmethod
    def setup(self) -> typing.Tuple[Figure, typing.List[Line]]:
        pass

    @abstractmethod
    def plot(self, metrics: typing.List[typing.Dict[str, np.ndarray]], plot: Figure, doc: Document):
        pass
