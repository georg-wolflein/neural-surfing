import itertools
import tensorflow as tf
from abc import ABC, abstractmethod


def scale_to_length(x: tf.Tensor, length: float) -> tf.Tensor:
    magnitude = tf.norm(x, axis=-1, keepdims=True)
    return (length / magnitude) * x


class SamplingTechnique(ABC):
    @abstractmethod
    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        pass


class ExhaustiveSamplingTechnique(SamplingTechnique):
    def __init__(self, num_weights: int, learning_rate: float, uniform_radius: bool = True):
        weight_changes = itertools.product([-1, 0, 1], repeat=num_weights)
        # Remove the entry with all zeros
        weight_changes = filter(lambda x: not all(
            map(int(0).__eq__, x)), weight_changes)
        weight_changes = tf.constant(list(weight_changes), dtype=tf.float32)
        if uniform_radius:
            weight_changes = scale_to_length(weight_changes, learning_rate)
        else:
            weight_changes *= learning_rate
        self.weight_changes = weight_changes

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        return weights + self.weight_changes


class RandomSamplingTechnique(SamplingTechnique):
    def __init__(self, num_weights: int, learning_rate: float, num_samples: int, uniform_radius: bool = True):
        self.num_weights = num_weights
        self.learning_rate = learning_rate
        self.num_samples = num_samples
        self.uniform_radius = uniform_radius

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        # Get random samples in the interval (0, 1]
        weight_changes = 1 - \
            tf.random.uniform(
                shape=(self.num_samples, self.num_weights), minval=0, maxval=1, dtype=tf.float32)
        # Make some of these samples negative, so we get the range [-1,1], but excluding 0
        weight_changes *= tf.cast(tf.random.uniform(shape=(self.num_samples,
                                                           self.num_weights),
                                                    minval=0, maxval=2, dtype=tf.int32) * 2 - 1,
                                  tf.float32)
        if self.uniform_radius:
            weight_changes = scale_to_length(
                weight_changes, self.learning_rate)
        else:
            weight_changes *= self.learning_rate
        return weights + weight_changes


RandomSamplingTechnique(2, .1, 5)([1, 2])
