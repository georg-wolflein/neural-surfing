"""A histogram plot.
"""

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
    """Class implementing the histogram visualisation.

    Histograms show the progression of one metric over time (epochs).
    """

    def __init__(self, metric: str, title: str = None):
        """Constructor.

        Arguments:
            metric {str} -- the name of the metric to visualise

        Keyword Arguments:
            title {str} -- optional title of the graph (default: {None})
        """

        super().__init__(x="epoch", y=metric,
                         title=title if title is not None else f"{metric} over time")
