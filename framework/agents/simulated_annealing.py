import tensorflow as tf
import numpy as np
from functools import partial
import typing

from problems import Problem
from .agent import GradientFreeAgent
from .util import get_num_weights
from.sampling import RandomSamplingGenerator


class SimulatedAnnealing(GradientFreeAgent):

    def __init__(self, problem: Problem, learning_rate: float, max_attempts_per_iteration: int = 15, energy_coefficient: float = 10000., temperature: float = 10000., cooling_rate: float = .05):
        super().__init__(problem, RandomSamplingGenerator(
            sample_radius=learning_rate, uniform_radius=True))
        self.max_attempts_per_iteration = max_attempts_per_iteration
        self.energy_coefficient = energy_coefficient
        self.temperature = temperature
        self.cooling_rate = cooling_rate

    def cost(self, current_output: tf.Tensor, target_output: tf.Tensor) -> tf.Tensor:
        return tf.norm(current_output - target_output, axis=-1)

    def choose_best_weight_update(self, weight_samples: typing.Iterator[tf.Tensor], weight_history: tf.Tensor, output_history: tf.Tensor, X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        current_cost = self.cost(output_history[-1], y)
        current_cost = tf.cast(current_cost, tf.float32)

        fallback = weight_history[-1]

        for _, weight_sample in zip(range(self.max_attempts_per_iteration), weight_samples):
            cost = self.cost(tf.reshape(
                self.predict_for_weights(weight_sample, X), y.shape), y)

            if cost < current_cost:
                accepted = True
            elif cost == current_cost:
                fallback = weight_sample
                continue
            else:
                # Calculate probability of accepting suboptimal state
                probability = tf.exp(self.energy_coefficient *
                                     (current_cost - cost) / self.temperature)
                accepted = probability > tf.random.uniform(tuple())

            if accepted:
                self.temperature *= 1 - self.cooling_rate
                return weight_sample

        return fallback
