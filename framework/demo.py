import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from agents.mse import MSE
from agents.loss_with_goal_line_deviation import LossWithGoalLineDeviation
from agents.greedy_probing import GreedyProbing
from agents.simulated_annealing import SimulatedAnnealing
from agents.sampling import ExhaustiveSamplingTechnique, RandomSamplingTechnique
from problems.stripe_problem import StripeProblem

from experiment import Experiment
from experiment.visualisations import Scatter2D, Scatter3D, Histogram

p = StripeProblem
learning_rate = 0.01
agents = {
    "MSE": MSE(p(), learning_rate),
    "Loss with goal line deviation": LossWithGoalLineDeviation(p(), learning_rate),
    "Greedy probing (random sampling)": GreedyProbing(p(), RandomSamplingTechnique(learning_rate=.01, num_samples=4)),
    "Greedy probing (exhaustive sampling)": GreedyProbing(p(), ExhaustiveSamplingTechnique(learning_rate=.01)),
    "Simulated annealing (random sampling)": SimulatedAnnealing(p(), RandomSamplingTechnique(learning_rate=.01, num_samples=4)),
    "Simulated annealing (exhaustive sampling)": SimulatedAnnealing(p(), ExhaustiveSamplingTechnique(learning_rate=.01))
}

experiment = Experiment(agents)
experiment.run([
    Scatter2D("weights"),
    Scatter2D("output:1", "output:3"),
    Histogram("output:0"),
    Histogram("output:1"),
    Histogram("output:2"),
    Histogram("output:3")], epoch_batch_size=10)
