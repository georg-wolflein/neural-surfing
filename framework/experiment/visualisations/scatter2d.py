import matplotlib.pyplot as plt
import typing
import numpy as np
from bokeh.plotting import Figure, figure
from bokeh.models import ColumnDataSource
from bokeh.models.glyphs import Line
from bokeh.document import Document
from tornado import gen
from functools import partial

from .visualisation import Visualisation
from ..metrics import Metric


class Scatter2D(Visualisation):
    """A two-dimensional scatter plot visualisation.
    """

    def __init__(self, x: str, y: str = None, title: str = None, x_title: str = None, y_title: str = None):
        """Constructor.

        Arguments:
            x {str} -- the x-axis metric (specified using the metric syntax)

        Keyword Arguments:
            y {str} -- the y-axis metric (if unspecified, the y axis will be the next dimension of the x-axis metric) (default: {None})
            title {str} -- optional title of the graph (default: {None})
            x_title {str} -- optional x-axis title (default: {None})
            y_title {str} -- optional y-axis title (default: {None})

        Raises:
            ValueError: if the metrics are not specified correctly
        """

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

    def setup(self, agents: typing.List[str], palette: list) -> typing.Tuple[Figure, typing.List[Line]]:
        # Override the setup method to initialize the graph in bokeh

        # Set the data source
        self._source = ColumnDataSource(data={
            **{f"x{i}": [] for i in range(len(agents))},
            **{f"y{i}": [] for i in range(len(agents))}
        })

        # Generate on line plot per agent
        plot = figure(title=self.title,
                      x_axis_label=self.x_title,
                      y_axis_label=self.y_title)
        lines = [plot.line(x=f"x{i}", y=f"y{i}",
                           source=self._source,
                           color=palette[i],
                           legend_label=agent)
                 for i, agent in enumerate(agents)]
        return plot, lines

    @gen.coroutine
    def _update(self, **kwargs):
        """Method to update the data.
        """
        self._source.stream(dict(**kwargs))

    def plot(self, metrics: typing.List[typing.Dict[str, np.ndarray]], plot: Figure, doc: Document):
        # Override the plot method to perform plotting

        def get_metrics() -> typing.Iterable[typing.Tuple[str, np.ndarray]]:
            """Method to get the metrics in the required format

            Yields:
                typing.Tuple[str, np.ndarray] -- the metric for a specific axis
            """
            for i, agent_metrics in enumerate(metrics):
                for axis, metric in zip(("x", "y"), self._metrics):
                    yield f"{axis}{i}", metric.select(agent_metrics)

        # Add a callback that will update the plot with the new data on the next tick
        doc.add_next_tick_callback(partial(self._source.stream,
                                           dict(get_metrics())))
