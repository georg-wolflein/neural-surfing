import matplotlib.pyplot as plt
import typing
import numpy as np
from bokeh.plotting import Figure, figure
from bokeh.models import ColumnDataSource
from bokeh.document import Document
from tornado import gen
from functools import partial

from .scatter2d import Scatter2D
from ..metrics import Metric


class Histogram(Scatter2D):

    def __init__(self, metric: str, title: str = None):
        super().__init__(x="epoch", y=metric,
                         title=title if title is not None else f"{metric} over time")
