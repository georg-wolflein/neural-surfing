import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from agents.mse import MSE
from agents.loss_with_goal_line_deviation import LossWithGoalLineDeviation
from agents.greedy_probing import GreedyProbing
from agents.sampling import ExhaustiveSamplingTechnique, RandomSamplingTechnique
from problems.simple_problem import SimpleProblem

from experiment import Experiment
from experiment.visualisations import Scatter2D, Scatter3D, Histogram

p = SimpleProblem
agents = {
    "MSE": MSE(p()),
    "Loss with goal line deviation": LossWithGoalLineDeviation(p()),
    "Greedy probing (random sampling)": GreedyProbing(p(), RandomSamplingTechnique(learning_rate=.01, num_samples=10)),
    "Greedy probing (exhaustive sampling)": GreedyProbing(p(), ExhaustiveSamplingTechnique(learning_rate=.01))
}

experiment = Experiment(agents)
experiment.run([
    Scatter2D("weights"),
    Scatter2D("output:1", "output:3"),
    Histogram("output:0"),
    Histogram("output:1"),
    Histogram("output:2"),
    Histogram("output:3")], epoch_batch_size=10)
