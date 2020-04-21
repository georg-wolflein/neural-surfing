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
    """Abstract class representing a neural agent.
    """

    def __init__(self, problem: Problem):
        """Constructor.

        Arguments:
            problem {Problem} -- the instance of the problem that this agent will train on
        """
        self.problem = problem

    @abstractmethod
    def compile(self):
        """Set up the agent.

        This method is called once before training and must be overridden
        """

    @abstractmethod
    def fit(self, X: tf.Tensor, y: tf.Tensor, epochs: int, callbacks: typing.List[tf.keras.callbacks.Callback]):
        """Abstract method to perform training. Akin to keras' fit method: the aim is to fit the model to the dataset.

        Arguments:
            X {tf.Tensor} -- input matrix
            y {tf.Tensor} -- output samples
            epochs {int} -- number of epochs to train for
            callbacks {typing.List[tf.keras.callbacks.Callback]} -- list of keras callbacks to be registered
        """

    def train(self, epochs: int, metrics: typing.List[str] = None) -> typing.Dict[str, np.ndarray]:
        """Train the agent for a specific number of epochs, logging the required metrics.

        Arguments:
            epochs {int} -- the number of epochs to train for

        Keyword Arguments:
            metrics {typing.List[str]} -- the epochs to log (need to be specified as part of the Problem instance) (default: {None})

        Returns:
            typing.Dict[str, np.ndarray] -- dictionary of collected metrics indexed by their name
        """

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
    """An abstract class representing an agent that uses derivatives (i.e. can train normally using the keras fit method).
    """

    def fit(self, *args, **kwargs):
        self.problem.model.fit(*args, **kwargs)


class GradientFreeAgent(Agent, ABC):
    """An abstract class representing a derivative-free agent. This means that the agent uses a sampling technique instead of relying on derivative information.
    """

    def __init__(self, problem: Problem, sampler: SamplingTechnique):
        """Constructor

        Arguments:
            problem {Problem} -- the instance of the problem that this agent will train on
            sampler {SamplingTechnique} -- the sampling technique to employ
        """

        super().__init__(problem)

        # Keep a record of the shape of the weights; will be required later to get and set weights
        self.weights_shape = list(
            map(tf.shape, self.problem.model.get_weights()))
        self.num_weights = get_num_weights(self.problem.model)

        # Initialize the sampler
        self.sampler = sampler
        sampler.initialize(self.num_weights)

    def get_weights(self) -> tf.Tensor:
        """Get the current weight state as a concatenated vector.

        Returns:
            tf.Tensor -- the weight vector
        """

        return tf.concat([tf.reshape(x, [-1]) for x in self.problem.model.get_weights()], axis=0)

    def set_weights(self, weights: tf.Tensor):
        """Set the weight vector.

        Internally, the weights are formed back into the shape required by keras.

        Arguments:
            weights {tf.Tensor} -- the weight vector
        """

        weights = tf.split(weights, list(
            map(tf.reduce_prod, self.weights_shape)))
        weights = [tf.reshape(x, shape)
                   for (x, shape) in zip(weights, self.weights_shape)]
        self.problem.model.set_weights(weights)

    def predict_for_weights(self, weights: tf.Tensor, X: tf.Tensor) -> tf.Tensor:
        """Get the prediction output for the model given a specific weight state.

        Note: this function does *not* change the weights back!

        Arguments:
            weights {tf.Tensor} -- the weight vector
            X {tf.Tensor} -- the input matrix

        Returns:
            tf.Tensor -- the outputs (predictions)
        """

        self.set_weights(weights)
        return self.problem.model.predict(X)

    def predict_for_multiple_weights(self, weights: tf.Tensor, X: tf.Tensor) -> tf.Tensor:
        """A batched version of the predict_for_weights function. 

        Batching is performed over the first dimension of the weights tensor.

        Arguments:
            weights {tf.Tensor} -- the weight matrix (collection of weight vectors)
            X {tf.Tensor} -- the input samples

        Returns:
            tf.Tensor -- the outputs for each weight vector
        """
        outputs = tf.map_fn(functools.partial(self.predict_for_weights, X=X),
                            weights,
                            parallel_iterations=1,
                            back_prop=False)
        return tf.reshape(outputs, (weights.shape[0], -1))

    def compile(self):
        # We can safely override this method to no-op because we require no initialisation
        pass

    @abstractmethod
    def choose_best_weight_update(self, weight_samples: tf.Tensor, weight_history: tf.Tensor, output_history: tf.Tensor, X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Abstract method that decides which weight update to choose.

        This is the only method that needs to be overridden by the agent implementation.
        Given a set of samples in weight space (produced by the sampling techniques) as well as other information (history of weights and outputs, as well as input samples and output targets),
        this function should return the new weight state that should be chosen.
        This method is called once per epoch.

        Arguments:
            weight_samples {tf.Tensor} -- the samples in weight space produced by the sampling technique
            weight_history {tf.Tensor} -- the history of weight states
            output_history {tf.Tensor} -- the history of predictions (output states)
            X {tf.Tensor} -- the input matrix
            y {tf.Tensor} -- the output targets

        Returns:
            tf.Tensor -- the new weight state chosen by the agent
        """

    def fit(self, X: tf.Tensor, y: tf.Tensor, epochs: int, callbacks: typing.List[tf.keras.callbacks.Callback]):
        # We provide a custom implementation of the fit method here, to make it easier for gradient-free agents to be implemented.
        # This method takes care of all the administrative tasks such as registering callbacks.
        # The user will only need to override the choose_best_weight_update method because this function will call it once per epoch to determine the new weight state.

        # Configure keras callbacks
        callbacks = configure_callbacks(callbacks, self.problem.model,
                                        epochs=epochs)
        callbacks._call_begin_hook(ModeKeys.TRAIN)

        # Keep a record of the weight and output history.
        # We will use numpy arrays instead of Python lists for performance, since we know the length of the history beforehand.
        # W cannot use the TensorFlow tensors themselves here because their values will change, but we want to record the *historical* values, hence we will be using numpy arrays.
        weight_history = np.zeros((epochs, self.num_weights))
        output_history = np.zeros((epochs, *y.shape))

        # Training loop (iterate over the epochs)
        for epoch in range(epochs):

            # If one of the keras callbacks caused early stopping, we will exit the training loop
            if callbacks.model.stop_training:
                break

            # Register the beginning of the epoch
            callbacks.on_epoch_begin(epoch, {})

            # Record the current weights and outputs
            weights = self.get_weights()
            outputs = self.problem.model.predict(X)
            weight_history[epoch] = weights
            output_history[epoch] = np.reshape(outputs, y.shape)

            # Get the weight samples from the sampler
            weight_samples = self.sampler(self.get_weights())

            # Consult the agent implementation for the best weight state to choose
            new_weights = self.choose_best_weight_update(weight_samples,
                                                         weight_history[:epoch+1],
                                                         output_history[:epoch+1],
                                                         X, y)

            # Set the weights
            self.set_weights(new_weights)

            # Signify end of epoch
            callbacks.on_epoch_end(epoch, {})

        # Signify end of training
        callbacks._call_end_hook(ModeKeys.TRAIN)
