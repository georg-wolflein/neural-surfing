
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import typing
import itertools

from agents.loss_with_goal_line_deviation import LossWithGoalLineDeviation
from agents import Agent
from problems import Problem
from .visualisations import Visualisation


class Experiment:

    def __init__(self, problem_factory: typing.Callable[[], Problem], agent_factories: typing.Sequence[typing.Callable[[Problem], Agent]]):
        self.agent_factories = agent_factories
        self.agents = [factory(problem_factory())
                       for factory in agent_factories]
        self.colors = cm.rainbow(np.linspace(0, 1, len(agent_factories)))[
            :, np.newaxis, :]

    def run(self, visualisations: typing.Sequence[Visualisation], epoch_batches: int = 100, epoch_batch_size: int = 50):
        metrics = set(itertools.chain(*[visualisation.required_metrics
                                        for visualisation in visualisations]))
        fig: plt.Figure = plt.figure()

        # Add legend
        fig.legend(handles=[mpatches.Patch(color=color, label=factory.__name__)
                            for (color, factory) in zip(np.squeeze(self.colors), self.agent_factories)])

        # Create subplots
        axes = [fig.add_subplot(1, len(visualisations), i, **v.subplot_kwargs)
                for i, v in enumerate(visualisations, 1)]
        for visualisation, ax in zip(visualisations, axes):
            visualisation.setup(ax)

        for _ in range(epoch_batches):
            for agent, color in zip(self.agents, self.colors):
                _, data = agent.train(epochs=epoch_batch_size,
                                      metrics=metrics)
                for visualisation, ax in zip(visualisations, axes):
                    visualisation.plot(data, color, ax)

            plt.draw()
            plt.pause(0.001)

        plt.show()
