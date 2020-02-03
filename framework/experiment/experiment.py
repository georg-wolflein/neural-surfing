
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import typing

from agents.loss_with_goal_line_deviation import LossWithGoalLineDeviation
from agents import Agent
from problems import Problem
from .visualisations import Visualisation
import itertools


class Experiment:

    def __init__(self, problem_factory: typing.Callable[[], Problem], agent_factories: typing.Sequence[typing.Callable[[Problem], Agent]]):
        self.agent_factories = agent_factories
        self.agents = [factory(problem_factory())
                       for factory in agent_factories]
        self.colors = cm.rainbow(np.linspace(0, 1, len(agent_factories)))[
            :, np.newaxis, :]

    def run(self, visualisations: typing.Sequence[Visualisation]):
        metrics = set(itertools.chain(*[visualisation.required_metrics
                                        for visualisation in visualisations]))
        fig = plt.figure()
        axes = [fig.add_subplot(1, len(visualisations), i, **v.subplot_kwargs)
                for i, v in enumerate(visualisations, 1)]
        for visualisation, ax in zip(visualisations, axes):
            visualisation.setup(ax, [factory.__name__
                                     for factory in self.agent_factories], self.colors)

        for i in range(100):
            for agent, color in zip(self.agents, self.colors):
                history, data = agent.train(epochs=50, metrics=metrics)
                for visualisation, ax in zip(visualisations, axes):
                    visualisation.plot(data, color, ax)

            plt.draw()
            plt.pause(0.001)

        plt.show()
