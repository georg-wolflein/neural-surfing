"""Demo for the shallow problem (see Figure 7 in ../progress/main.pdf).
"""

from nf.agents.mse import MSE
from nf.agents.loss_with_goal_line_deviation import LossWithGoalLineDeviation
from nf.agents.greedy_probing import GreedyProbing
from nf.agents.simulated_annealing import SimulatedAnnealing
from nf.agents.sampling import ExhaustiveSamplingTechnique, RandomSamplingTechnique
from nf.problems.shallow_problem import ShallowProblem

from nf.experiment import Experiment
from nf.experiment.visualisations.scatter2d import Scatter2D
from nf.experiment.visualisations.histogram import Histogram
from util import get_demo_args

# Get command-line arguments
args = get_demo_args()

# Initialise problem and agents
p = ShallowProblem
learning_rate = 0.01
agents = {
    "MSE": MSE(p(), learning_rate*10),
    "Loss with goal line deviation": LossWithGoalLineDeviation(p(), learning_rate*10),
    "Greedy probing (random sampling)": GreedyProbing(p(), RandomSamplingTechnique(learning_rate, num_samples=4)),
    "Greedy probing (exhaustive sampling)": GreedyProbing(p(), ExhaustiveSamplingTechnique(learning_rate)),
    "Simulated annealing": SimulatedAnnealing(p(), learning_rate)
}

# Run the experiment with the visualisations
experiment = Experiment(agents)
experiment.run_server([
    Scatter2D("weights"),
    Histogram("loss"),
    Histogram("output:0"),
    Histogram("output:1"),
    Histogram("output:2"),
    Histogram("output:3"),
    Scatter2D("output:0", "output:1"),
    Scatter2D("output:2", "output:3"),
    Histogram("run_time")
], title="Shallow problem", **args)
