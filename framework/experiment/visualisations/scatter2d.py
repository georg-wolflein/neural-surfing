import matplotlib.pyplot as plt
import typing
import numpy as np
from bokeh.plotting import Figure, figure
from bokeh.models import ColumnDataSource
from bokeh.document import Document
from tornado import gen
from functools import partial

from .visualisation import Visualisation
from ..metrics import Metric


class Scatter2D(Visualisation):

    def __init__(self, x: str, y: str = None, title: str = None, x_title: str = None, y_title: str = None):

        if y is None:
            if ":" not in x:
                y = x + ":1"
                x += ":0"
            else:
                raise ValueError("missing y metric")

        self._metrics = list(map(Metric.from_string, (x, y)))
        required_metrics = {m.name for m in self._metrics}

        self.title = title if title is not None else " vs ".join(
            required_metrics)
        self.x_title = x_title if x_title is not None else str(
            self._metrics[0])
        self.y_title = y_title if y_title is not None else str(
            self._metrics[1])

        super().__init__(required_metrics)

    def setup(self, num_agents: int, palette: list) -> Figure:
        self._source = ColumnDataSource(data={
            **{f"x{i}": [] for i in range(num_agents)},
            **{f"y{i}": [] for i in range(num_agents)}
        })
        plot = figure(title=self.title,
                      x_axis_label=self.x_title,
                      y_axis_label=self.y_title)
        for i in range(num_agents):
            plot.line(x=f"x{i}", y=f"y{i}",
                      source=self._source,
                      color=palette[i])
        return plot

    @gen.coroutine
    def _update(self, **kwargs):
        self._source.stream(dict(**kwargs))

    def plot(self, metrics: typing.List[typing.Dict[str, np.ndarray]], plot: Figure, doc: Document):
        def get_metrics():
            for i, agent_metrics in enumerate(metrics):
                for axis, metric in zip(("x", "y"), self._metrics):
                    yield f"{axis}{i}", metric.select(agent_metrics)

        doc.add_next_tick_callback(partial(self._source.stream,
                                           dict(get_metrics())))
