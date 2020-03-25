import itertools
import tensorflow as tf
from abc import ABC, abstractmethod
import typing


def scale_to_length(x: tf.Tensor, length: float) -> tf.Tensor:
    magnitude = tf.norm(x, axis=-1, keepdims=True)
    return (length / magnitude) * x


class SamplingTechnique(ABC):

    @abstractmethod
    def initialize(self, num_weights: int):
        pass

    @abstractmethod
    def __call__(self, weights: tf.Tensor) -> typing.Union[tf.Tensor, typing.Iterator[tf.Tensor]]:
        pass


class ExhaustiveSamplingTechnique(SamplingTechnique):
    def __init__(self, sample_radius: float, uniform_radius: bool = True):
        self.sample_radius = sample_radius
        self.uniform_radius = uniform_radius

    def initialize(self, num_weights: int):
        weight_changes = itertools.product([-1, 0, 1], repeat=num_weights)
        # Remove the entry with all zeros
        weight_changes = filter(lambda x: not all(
            map(int(0).__eq__, x)), weight_changes)
        weight_changes = tf.constant(list(weight_changes), dtype=tf.float32)
        if self.uniform_radius:
            weight_changes = scale_to_length(
                weight_changes, self.sample_radius)
        else:
            weight_changes *= self.sample_radius
        self.weight_changes = weight_changes

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        return weights + self.weight_changes


class RandomSamplingTechnique(SamplingTechnique):
    def __init__(self, sample_radius: float, num_samples: int, uniform_radius: bool = True):
        self.sample_radius = sample_radius
        self.num_samples = num_samples
        self.uniform_radius = uniform_radius

    def initialize(self, num_weights: int):
        self.num_weights = num_weights

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        # Get random samples in the interval (0, 1]
        weight_changes = 1 - \
            tf.random.uniform(
                shape=(self.num_samples, self.num_weights), minval=0, maxval=1, dtype=tf.float32)
        # Make some of these samples negative, so we get the range [-1,1], but excluding 0
        sign = tf.sign(weight_changes - 0.5)
        weight_changes *= tf.where(sign != 0., sign, 1.)
        if self.uniform_radius:
            weight_changes = scale_to_length(
                weight_changes, self.sample_radius)
        else:
            weight_changes *= self.sample_radius
        return weights + weight_changes


class RandomSamplingGenerator(SamplingTechnique):
    def __init__(self, sample_radius: float, uniform_radius: bool = True):
        self.sample_radius = sample_radius
        self.uniform_radius = uniform_radius

    def initialize(self, num_weights: int):
        self.num_weights = num_weights

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        while True:
            # Get random samples in the interval (0, 1]
            weight_changes = 1 - tf.random.uniform(shape=(self.num_weights,),
                                                   minval=0, maxval=1, dtype=tf.float32)
            # Make some of these samples negative, so we get the range [-1,1], but excluding 0
            sign = tf.sign(weight_changes - 0.5)
            weight_changes *= tf.where(sign == 0., 1., sign)

            if self.uniform_radius:
                weight_changes = scale_to_length(
                    weight_changes, self.sample_radius)
            else:
                weight_changes *= self.sample_radius
            yield weights + weight_changes
