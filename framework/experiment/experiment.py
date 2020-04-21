
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import typing
import itertools
from threading import Thread
import time
from bokeh.plotting import Document
from bokeh.layouts import gridplot, row, column
from bokeh.palettes import Category10
from bokeh.models import CheckboxButtonGroup
from bokeh.models.callbacks import CustomJS
from bokeh.server.server import Server

from agents.loss_with_goal_line_deviation import LossWithGoalLineDeviation
from agents import Agent
from problems import Problem
from .visualisations import Visualisation


class Experiment:
    """Class representing an experiment that can be run.

    When it is run, the experment will coordinate the training of the agents. 
    The experiment will also manage the collection and aggregation of metrics, and ensure data is passed to the visualisations to update them in real time.
    """

    def __init__(self, agents: typing.Dict[str, Agent]):
        """Constructor.

        Arguments:
            agents {typing.Dict[str, Agent]} -- the list of agents that will be run for this experiment
        """

        self.agent_names = list(agents.keys())
        self.agents = list(agents.values())
        self.colors = cm.rainbow(np.linspace(0, 1, len(agents)))[
            :, np.newaxis, :]

    def run(self, doc: Document, visualisations: typing.Sequence[Visualisation], epoch_batches: int = 100, epoch_batch_size: int = 50, cols: int = 2, title: str = "Experiment"):
        """Run the experiment.

        Arguments:
            doc {Document} -- the bokeh Document
            visualisations {typing.Sequence[Visualisation]} -- the visualisations to show in real time

        Keyword Arguments:
            epoch_batches {int} -- the number of batches of training to perform for each agent (default: {100})
            epoch_batch_size {int} -- the number of epochs to train for per training batch (default: {50})
            cols {int} -- the number of columns to display the visualisations in (default: {2})
            title {str} -- optional title of the web page (default: {"Experiment"})
        """

        # Determine which metrics need to be calculated.
        # This will ensure that we do not calculate useless metrics that are not visualised.
        metrics = set(itertools.chain(*[visualisation.required_metrics
                                        for visualisation in visualisations]))

        # Create plots
        plots, lines = zip(*[visualisation.setup(self.agent_names, palette=Category10[10])
                             for visualisation in visualisations])

        # Create a button per agent to toggle their visibility
        buttons = CheckboxButtonGroup(
            labels=self.agent_names,
            active=list(range(len(self.agents))))
        buttons.callback = CustomJS(args=dict(buttons=buttons, lines=lines),
                                    code="""console.log(buttons);
                                    lines.forEach(plot => plot.forEach((line, index) => {
                                        line.visible = buttons.active.includes(
                                            index);
                                    }));
                                    """)

        # Add the title, buttons, and plots
        doc.title = title
        doc.add_root(column(buttons,
                            gridplot(plots, ncols=cols)))

        def run_blocking():
            """Method to run the training of each agent that will update the visualisations in real time.

            This method should run on a separate thread.
            """

            # Compile the agents
            [agent.compile() for agent in self.agents]

            # Plot initial point
            agent_data = [{**{k: v[np.newaxis, ...]
                              for(k, v)in agent.problem.evaluate_metrics(metrics=metrics).items()},
                           "epoch": np.array([0]),
                           "run_time": np.array([0])}
                          for agent in self.agents]
            for visualisation, plot in zip(visualisations, plots):
                visualisation.plot(agent_data, plot, doc)

            # Perform training and continually plot
            for epoch_batch in range(epoch_batches):
                start_epoch = epoch_batch * epoch_batch_size + 1

                def get_agent_data():
                    for agent in self.agents:
                        start = time.perf_counter()
                        data = agent.train(epochs=epoch_batch_size,
                                           metrics=metrics)
                        end = time.perf_counter()
                        data["epoch"] = np.arange(
                            start_epoch, start_epoch + epoch_batch_size)
                        data["run_time"] = np.repeat(
                            end-start, epoch_batch_size) / epoch_batch_size
                        yield data
                data = list(get_agent_data())
                for visualisation, plot in zip(visualisations, plots):
                    visualisation.plot(data, plot, doc)

        # Run the experiment in a separate thread
        thread = Thread(target=run_blocking)
        thread.start()

    def run_server(self, *args, port: int = 5000, **kwargs):
        """Start the bokeh server with the experiment (*args and **kwargs are passed on to the Experiment.run() method).

        Keyword Arguments:
            port {int} -- the port to run the server on (default: {5000})
        """
        server = Server({"/": lambda doc: self.run(doc, *args, **kwargs)},
                        port=port,
                        num_procs=1)
        server.start()

        print(f"Starting experiment at http://localhost:{port}/")
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()
