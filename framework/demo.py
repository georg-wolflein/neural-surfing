import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from agents.loss_with_goal_line_deviation import LossWithGoalLineDeviation
from agents.mse import MSE
from problems.simple_problem import SimpleProblem

from experiment import Experiment
from experiment.visualisations import Scatter2D, Scatter3D, Histogram

problem_factory = SimpleProblem
agent_factories = [LossWithGoalLineDeviation, MSE]

experiment = Experiment(problem_factory, agent_factories)
experiment.run([
    Scatter2D("weights"),
    Scatter2D("output:1", "output:3"),
    Histogram("output:0"),
    Histogram("output:1"),
    Histogram("output:2"),
    Histogram("output:3")])
