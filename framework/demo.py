import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from agents.loss_with_goal_line_deviation import LossWithGoalLineDeviation
from agents.mse import MSE
from problems.shallow_problem import ShallowProblem

y = np.array([
    .2,
    .8,
    .8, .8, .8
])
x = np.array([
    (.6, .4),
    (1., .4),
    (1., .4), (1., .4), (1., .4)
])

agent_factories = [LossWithGoalLineDeviation, MSE]
agents = [agent(ShallowProblem()) for agent in agent_factories]
colors = cm.rainbow(np.linspace(0, 1, len(agents)))[:, np.newaxis, :]

for factory, color in zip(agent_factories, colors):
    plt.scatter([], [], c=color, label=factory.__name__)

plt.legend()

for i in range(10):
    for agent, color in zip(agents, colors):
        history, data = agent.train(x, y, epochs=10)
        weights = data["weights"]
        plt.scatter(*weights.T, c=color)

    plt.draw()
    plt.pause(0.001)

plt.show()
