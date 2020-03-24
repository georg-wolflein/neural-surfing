
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import typing
import itertools
from threading import Thread
from bokeh.plotting import curdoc
from bokeh.layouts import gridplot, row, column
from bokeh.palettes import Category10
from bokeh.models import CheckboxButtonGroup
from bokeh.models.callbacks import CustomJS

from agents.loss_with_goal_line_deviation import LossWithGoalLineDeviation
from agents import Agent
from problems import Problem
from .visualisations import Visualisation


class Experiment:

    def __init__(self, agents: typing.Dict[str, Agent]):
        self.agent_names = list(agents.keys())
        self.agents = list(agents.values())
        self.colors = cm.rainbow(np.linspace(0, 1, len(agents)))[
            :, np.newaxis, :]

    def run(self, visualisations: typing.Sequence[Visualisation], epoch_batches: int = 100, epoch_batch_size: int = 50, cols: int = 2, title: str = "Experiment"):
        metrics = set(itertools.chain(*[visualisation.required_metrics
                                        for visualisation in visualisations]))

        # Create plots
        plots, lines = zip(*[visualisation.setup(len(self.agents), palette=Category10[10])
                             for visualisation in visualisations])

        buttons = CheckboxButtonGroup(
            labels=self.agent_names,
            active=list(range(len(self.agents))))

        buttons.callback = CustomJS(args=dict(buttons=buttons, lines=lines),
                                    code="""
                                    lines.forEach(plot => plot.forEach((line, index) => {
                                        line.visible = buttons.active.includes(
                                            index);
                                    }));
                                    """)

        doc = curdoc()
        doc.title = title
        doc.add_root(column(buttons,
                            gridplot(plots, ncols=cols)))

        def run_blocking():

            # Compile the agents
            [agent.compile() for agent in self.agents]

            # Plot initial point
            agent_data = [{**{k: v[np.newaxis, ...]
                              for(k, v)in agent.problem.evaluate_metrics(metrics=metrics).items()},
                           "epoch": np.array([0])}
                          for agent in self.agents]
            for visualisation, plot in zip(visualisations, plots):
                visualisation.plot(agent_data, plot, doc)

            # Perform training and continually plot
            for epoch_batch in range(epoch_batches):
                start_epoch = epoch_batch * epoch_batch_size + 1
                agent_data = [{**agent.train(epochs=epoch_batch_size, metrics=metrics),
                               "epoch": np.arange(start_epoch, start_epoch + epoch_batch_size)}
                              for agent in self.agents]
                for visualisation, plot in zip(visualisations, plots):
                    visualisation.plot(agent_data, plot, doc)

        thread = Thread(target=run_blocking)
        thread.start()
