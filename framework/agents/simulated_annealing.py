import tensorflow as tf
import numpy as np
from functools import partial
import typing

from problems import Problem
from .agent import SamplingBasedAgent
from .util import get_num_weights
from.sampling import SamplingTechnique


class SimulatedAnnealing(SamplingBasedAgent):

    def __init__(self, problem: Problem, sampler: SamplingTechnique, energy_coefficient: float = 10000000., temperature: float = 10000., cooling_rate: float = .1):
        super().__init__(problem, sampler)
        self.energy_coefficient = energy_coefficient
        self.temperature = temperature
        self.cooling_rate = cooling_rate

    def cost(self, current_output: tf.Tensor, target_output: tf.Tensor) -> tf.Tensor:
        return tf.norm(current_output - target_output, axis=-1)

    def choose_best_weight_update(self, weight_samples: tf.Tensor, output_samples: tf.Tensor, weight_history: tf.Tensor, output_history: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        costs = self.cost(output_samples, y)
        current_cost = self.cost(output_history[-1], y)
        current_cost = tf.cast(current_cost, tf.float32)

        # Calculate probabilities for accepting suboptimal state
        probability = tf.exp(self.energy_coefficient *
                             (current_cost - costs) / self.temperature)
        print(tf.boolean_mask(probability, current_cost < costs))

        accepted_moves = (probability > tf.random.uniform(
            probability.shape)) | (costs < current_cost)
        weight_samples = tf.boolean_mask(weight_samples, accepted_moves)

        # Randomly choose one of the options
        index = tf.random.uniform(
            shape=[], minval=0, maxval=weight_samples.shape[0], dtype=tf.int32)

        self.temperature *= 1 - self.cooling_rate
        return weight_samples[index]
