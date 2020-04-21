import tensorflow as tf
import numpy as np
from functools import partial
import typing

from problems import Problem
from .agent import GradientFreeAgent
from .util import get_num_weights
from.sampling import RandomSamplingGenerator


class SimulatedAnnealing(GradientFreeAgent):
    """Implementation of the simulated annealing agent.
    """

    def __init__(self, problem: Problem, learning_rate: float, max_attempts_per_iteration: int = 15, energy_coefficient: float = 10000., temperature: float = 10000., cooling_rate: float = .05):
        """Constructor.

        Arguments:
            problem {Problem} -- the instance of the problem that this agent will train on
            learning_rate {float} -- the learning rate (essentially the sampling radius)

        Keyword Arguments:
            max_attempts_per_iteration {int} -- the maximum number of non-accepting samples to evaluate each iteration before falling back the best one seen thus far (default: {15})
            energy_coefficient {float} -- the energy coefficient (default: {10000.})
            temperature {float} -- the initial temperature (default: {10000.})
            cooling_rate {float} -- the cooling rate (default: {.05})
        """

        super().__init__(problem, RandomSamplingGenerator(
            sample_radius=learning_rate, uniform_radius=True))
        self.max_attempts_per_iteration = max_attempts_per_iteration
        self.energy_coefficient = energy_coefficient
        self.temperature = temperature
        self.cooling_rate = cooling_rate

    def cost(self, current_output: tf.Tensor, target_output: tf.Tensor) -> tf.Tensor:
        """A simple cost function.

        Arguments:
            current_output {tf.Tensor} -- the current output
            target_output {tf.Tensor} -- the target

        Returns:
            tf.Tensor -- the Euclidean distance between the current and target output
        """

        return tf.norm(current_output - target_output, axis=-1)

    def choose_best_weight_update(self, weight_samples: typing.Iterator[tf.Tensor], weight_history: tf.Tensor, output_history: tf.Tensor, X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:

        # Calculate the current cost
        current_cost = self.cost(output_history[-1], y)
        current_cost = tf.cast(current_cost, tf.float32)

        # Remember the last weight state as fallback
        fallback = weight_history[-1]

        # Iterate over the samples, generating them on demand (for a maximum of max_attempts_per_iteration times)
        for _, weight_sample in zip(range(self.max_attempts_per_iteration), weight_samples):

            # Calculate the sample's cost
            cost = self.cost(tf.reshape(
                self.predict_for_weights(weight_sample, X), y.shape), y)

            # Determine if we should accept the sample
            if cost < current_cost:

                # If the samples cost is less then the current cost, accept
                accepted = True

            elif cost == current_cost:

                # If the current cost equals the sample cost, remember this sample for fallback
                fallback = weight_sample
                continue
            else:

                # Otherwise, calculate probability of accepting suboptimal state
                probability = tf.exp(self.energy_coefficient *
                                     (current_cost - cost) / self.temperature)
                # Perform probability test
                accepted = probability > tf.random.uniform(tuple())

            # If we have accepted the sample, cool down and return it
            if accepted:
                self.temperature *= 1 - self.cooling_rate
                return weight_sample

        # We reach this point when exceeding the maximum number of samples to be tested
        return fallback
