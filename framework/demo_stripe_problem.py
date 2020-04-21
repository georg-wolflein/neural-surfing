"""Demo of agents on the stripe problem.
"""

from agents import MSE, LossWithGoalLineDeviation, GreedyProbing, SimulatedAnnealing
from agents.sampling import ExhaustiveSamplingTechnique, RandomSamplingTechnique
from problems.stripe_problem import StripeProblem

from experiment import Experiment
from experiment.visualisations import Scatter2D, Histogram
from util import get_demo_args

# Get command-line arguments
args = get_demo_args()

# Initialise problem and agents
p = StripeProblem
learning_rate = 0.01
agents = {
    "MSE": MSE(p(), learning_rate*10),
    "Loss with goal line deviation": LossWithGoalLineDeviation(p(), learning_rate*10),
    "Greedy probing (random sampling)": GreedyProbing(p(), RandomSamplingTechnique(learning_rate, num_samples=4)),
    "Greedy probing (exhaustive sampling)": GreedyProbing(p(), ExhaustiveSamplingTechnique(learning_rate)),
    "Simulated annealing": SimulatedAnnealing(p(), learning_rate, cooling_rate=.1)
}

# Run the experiment with the visualisations
experiment = Experiment(agents)
experiment.run_server([
    Scatter2D("weights"),
    Histogram("loss"),
    Scatter2D("output:0", "output:1"),
    Scatter2D("output:0", "output:2"),
    Histogram("output:0"),
    Histogram("output:1"),
    Histogram("output:2"),
    Histogram("run_time")
], title="Stripe problem", **args)
