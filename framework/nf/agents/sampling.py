"""Sampling techniques for gradient-free agents.
"""

import itertools
import tensorflow as tf
from abc import ABC, abstractmethod
import typing

from .util import scale_to_length


class SamplingTechnique(ABC):
    """Abstract base class representing a sampling technique (for weight space).
    """

    @abstractmethod
    def initialize(self, num_weights: int):
        """Initialize the sampling technique.

        This method should be called once before training. 
        Its purpose is to pre-compute certain values such that they do not have to be computed over and over again during training.

        Arguments:
            num_weights {int} -- the dimensionality of the weight space
        """

    @abstractmethod
    def __call__(self, weights: tf.Tensor) -> typing.Union[tf.Tensor, typing.Iterator[tf.Tensor]]:
        """Obtain a sample given the current weight state.

        Arguments:
            weights {tf.Tensor} -- the current weight state

        Returns:
            typing.Union[tf.Tensor, typing.Iterator[tf.Tensor]] -- depending on the type of sampling technique, this will be a tensor of all weight samples or a generator to arbitrarily generate weight samples for computational efficiency.
        """


class ExhaustiveSamplingTechnique(SamplingTechnique):
    """Implementation of an exhaustive sampling technique that obtains samples with a specific radius along all possible 45° directions in a weight space of arbitrary dimensionality.

    The number of samples will be 3^N-1 where N is the dimensionality of the weight space.
    """

    def __init__(self, sample_radius: float, uniform_radius: bool = True):
        """Constructor.

        Arguments:
            sample_radius {float} -- the sample radius

        Keyword Arguments:
            uniform_radius {bool} -- whether or not to ensure that all samples have the same length (if False, the samples at 45° angles will be longer than the ones at 90° angles to any weight axis) (default: {True})
        """
        self.sample_radius = sample_radius
        self.uniform_radius = uniform_radius

    def initialize(self, num_weights: int):
        # We will use this method to pre-compute the weight changes, so they can just be added to the current weight state to get the samples later

        # First, we will get all combinations of [-1,0,1] of dimensionality N (number of weight)
        weight_changes = itertools.product([-1, 0, 1], repeat=num_weights)

        # Remove the entry with all zeros (that would just be the current weight state)
        weight_changes = filter(lambda x: not all(
            map(int(0).__eq__, x)), weight_changes)

        # Convert to TensorFlow constant
        weight_changes = tf.constant(list(weight_changes), dtype=tf.float32)

        # Scale weight changes
        if self.uniform_radius:
            weight_changes = scale_to_length(
                weight_changes, self.sample_radius)
        else:
            weight_changes *= self.sample_radius
        self.weight_changes = weight_changes

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        # Return the samples
        return weights + self.weight_changes


class RandomSamplingTechnique(SamplingTechnique):
    """Implementation of a random sampling technique that obtains samples within a specific radius in a weight space of arbitrary dimensionality.
    """

    def __init__(self, sample_radius: float, num_samples: int, uniform_radius: bool = True):
        """Constructor.

        Arguments:
            sample_radius {float} -- the sample radius
            num_samples {int} -- the number of random samples to generate

        Keyword Arguments:
            uniform_radius {bool} -- whether the samples should be along the circumference of the sampling circle (True) or in the area of the sampling circle (False) (default: {True})
        """
        self.sample_radius = sample_radius
        self.num_samples = num_samples
        self.uniform_radius = uniform_radius

    def initialize(self, num_weights: int):
        # Remember the number of weights
        self.num_weights = num_weights

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        # Get random samples in the interval (0, 1]
        weight_changes = 1 - \
            tf.random.uniform(
                shape=(self.num_samples, self.num_weights), minval=0, maxval=1, dtype=tf.float32)

        # Make some of these samples negative, so we get the range [-1,1], but excluding 0
        sign = tf.sign(weight_changes - 0.5)
        weight_changes *= tf.where(sign != 0., sign, 1.)

        # Scale the samples
        if self.uniform_radius:
            weight_changes = scale_to_length(
                weight_changes, self.sample_radius)
        else:
            weight_changes *= self.sample_radius
        return weights + weight_changes


class RandomSamplingGenerator(SamplingTechnique):
    """Similar to RandomSamplingTechnique, except that samples are obtained on-demand using a generator.

    This should be used when the number of samples needed is not known beforehand, and each sample is processed separately until reaching a termination condition.
    """

    def __init__(self, sample_radius: float, uniform_radius: bool = True):
        """Constructor.

        Arguments:
            sample_radius {float} -- the sample radius
            num_samples {int} -- the number of random samples to generate

        Keyword Arguments:
            uniform_radius {bool} -- whether the samples should be along the circumference of the sampling circle (True) or in the area of the sampling circle (False) (default: {True})
        """
        self.sample_radius = sample_radius
        self.uniform_radius = uniform_radius

    def initialize(self, num_weights: int):
        # Record the number of weights
        self.num_weights = num_weights

    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        # Generator loop
        while True:

            # Get random samples in the interval (0, 1]
            weight_changes = 1 - tf.random.uniform(shape=(self.num_weights,),
                                                   minval=0, maxval=1, dtype=tf.float32)
            # Make some of these samples negative, so we get the range [-1,1], but excluding 0
            sign = tf.sign(weight_changes - 0.5)
            weight_changes *= tf.where(sign == 0., 1., sign)

            # Scale the samples
            if self.uniform_radius:
                weight_changes = scale_to_length(
                    weight_changes, self.sample_radius)
            else:
                weight_changes *= self.sample_radius

            # Yield on demand
            yield weights + weight_changes
