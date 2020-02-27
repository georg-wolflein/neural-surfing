import matplotlib.pyplot as plt
import typing
import numpy as np
from bokeh.plotting import Figure, figure
from bokeh.models import ColumnDataSource
from bokeh.document import Document
from tornado import gen
from functools import partial

from .visualisation import Visualisation


class Scatter2D(Visualisation):

    def __init__(self, x: typing.Union[str, tuple], y: typing.Union[str, tuple] = None, title: str = None, x_title: str = None, y_title: str = None):

        def format_axis_metric(axis: typing.Union[str, tuple]) -> tuple:
            if isinstance(axis, str):
                if ":" in axis:
                    return tuple(fn(x) for (fn, x) in zip((str, int), axis.split(":")))
                else:
                    return axis, 0
            elif isinstance(axis, tuple):
                if len(axis) != 2:
                    raise ValueError()
                return axis
            else:
                raise ValueError()

        x_metric, x_dim = format_axis_metric(x)
        if y is None:
            y_metric, y_dim = x_metric, x_dim+1
        else:
            y_metric, y_dim = format_axis_metric(y)

        self._metrics = [(x_metric, x_dim), (y_metric, y_dim)]
        required_metrics = {x_metric, y_metric}

        self.title = title if title is not None else ", ".join(
            required_metrics)
        self.x_title = x_title if x_title is not None else ":".join(
            map(str, self._metrics[0]))
        self.y_title = y_title if y_title is not None else ":".join(
            map(str, self._metrics[1]))

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
                for axis, (axis_metric, axis_dimension) in zip(("x", "y"), self._metrics):
                    yield f"{axis}{i}", agent_metrics[axis_metric][..., axis_dimension]

        doc.add_next_tick_callback(partial(self._source.stream,
                                           dict(get_metrics())))
