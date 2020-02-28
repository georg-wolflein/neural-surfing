import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from agents.loss_with_goal_line_deviation import LossWithGoalLineDeviation
from agents.mse import MSE
from problems.shallow_problem import ShallowProblem

from experiment import Experiment
from experiment.visualisations import Scatter2D, Scatter3D, Histogram

problem_factory = ShallowProblem
agent_factories = [LossWithGoalLineDeviation, MSE]

experiment = Experiment(problem_factory, agent_factories)
experiment.run([
    Scatter2D("weights"),
    Scatter2D("output"),
    Scatter2D("weights:0", "output:1"),
    Scatter2D("epoch", "weights:0", title="w1 over time"),
    Histogram("weights:1")])
