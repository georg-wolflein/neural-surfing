from abc import ABC, abstractmethod
import typing
import functools
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import configure_callbacks
from tensorflow.python.keras.utils.mode_keys import ModeKeys

from problems import Problem
from .sampling import SamplingTechnique
from .util import get_num_weights


class Agent(ABC):

    def __init__(self, problem: Problem):
        self.problem = problem

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def fit(self, X: tf.Tensor, y: tf.Tensor, epochs: int, callbacks: typing.List[tf.keras.callbacks.Callback]):
        pass

    def train(self, epochs: int, metrics: typing.List[str] = None):
        data = {}

        def callback(epoch, logs={}):
            calculated_metrics = self.problem.evaluate_metrics(metrics=metrics)
            if len(data) == 0:
                data.update({
                    name: np.zeros(shape=(epochs, *value.shape),
                                   dtype=value.dtype)
                    for name, value in calculated_metrics.items()
                })
            for name, value in calculated_metrics.items():
                data[name][epoch] = value

        self.fit(self.problem.X, self.problem.y,
                 epochs=epochs,
                 callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=callback)])

        return data


class GradientBasedAgent(Agent, ABC):

    def fit(self, *args, **kwargs):
        self.problem.model.fit(*args, **kwargs)


class SamplingBasedAgent(Agent, ABC):
    def __init__(self, problem: Problem, sampler: SamplingTechnique):
        super().__init__(problem)
        self.sampler = sampler
        self.weights_shape = list(
            map(tf.shape, self.problem.model.get_weights()))
        self.num_weights = get_num_weights(self.problem.model)
        sampler.initialize(self.num_weights)

    def compile(self):
        pass

    @abstractmethod
    def choose_best_weight_update(self, weight_samples: tf.Tensor, output_samples: tf.Tensor, weight_history: tf.Tensor, output_history: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        pass

    def get_weights(self) -> tf.Tensor:
        return tf.concat([tf.reshape(x, [-1]) for x in self.problem.model.get_weights()], axis=0)

    def set_weights(self, weights: tf.Tensor):
        weights = tf.split(weights, list(
            map(tf.reduce_prod, self.weights_shape)))
        weights = [tf.reshape(x, shape)
                   for (x, shape) in zip(weights, self.weights_shape)]
        self.problem.model.set_weights(weights)

    def predict_for_weights(self, weights: tf.Tensor, X: tf.Tensor) -> tf.Tensor:
        self.set_weights(weights)
        return self.problem.model.predict(X)

    def fit(self, X: tf.Tensor, y: tf.Tensor, epochs: int, callbacks: typing.List[tf.keras.callbacks.Callback]):
        callbacks = configure_callbacks(callbacks, self.problem.model,
                                        epochs=epochs)
        callbacks._call_begin_hook(ModeKeys.TRAIN)

        weight_history = np.zeros((epochs, self.num_weights))
        output_history = np.zeros((epochs, *y.shape))

        for epoch in range(epochs):
            if callbacks.model.stop_training:
                break

            callbacks.on_epoch_begin(epoch, {})

            weight_samples = self.sampler(self.get_weights())
            num_samples = weight_samples.shape[0]
            output_samples = tf.map_fn(functools.partial(self.predict_for_weights, X=X),
                                       weight_samples,
                                       parallel_iterations=1,
                                       back_prop=False)
            output_samples = tf.reshape(output_samples, (num_samples, -1))
            new_weights = self.choose_best_weight_update(weight_samples, output_samples,
                                                         weight_history[:epoch], output_history[:epoch], y)
            weight_history[epoch] = new_weights
            output_history[epoch] = np.reshape(
                self.predict_for_weights(new_weights, X), y.shape)

            callbacks.on_epoch_end(epoch, {})

        callbacks._call_end_hook(ModeKeys.TRAIN)
