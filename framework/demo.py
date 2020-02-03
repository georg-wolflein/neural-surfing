import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from agents.loss_with_goal_line_deviation import LossWithGoalLineDeviation
from agents.mse import MSE
from problems.shallow_problem import ShallowProblem

from experiment import Experiment
from experiment.visualisations import Scatter2D, Scatter3D

problem_factory = ShallowProblem
agent_factories = [LossWithGoalLineDeviation, MSE]
colors = cm.rainbow(np.linspace(0, 1, len(agent_factories)))[:, np.newaxis, :]

experiment = Experiment(problem_factory, agent_factories)
experiment.run([
    Scatter2D("weights"),
    Scatter2D("output"),
    Scatter3D("output")])
