"""Utilities for the problems module.
"""

import tensorflow as tf


class DenseWithFixedBias(tf.keras.layers.Layer):
    """A fully-conected keras layer with a fixed bias term.
    """

    def __init__(self, num_outputs: int, bias: float, kernel: tf.Tensor):
        """Constructor.

        Arguments:
            num_outputs {int} -- number of output units
            bias {float} -- value of the bias
            kernel {tf.Tensor} -- the initial weights
        """
        super().__init__()
        self.num_outputs = num_outputs
        self.bias = bias
        self.kernel_initializer = tf.constant_initializer(kernel)

    def build(self, input_shape: tf.TensorShape):
        """Build the layer for keras.

        Arguments:
            input_shape {tf.TensorShape} -- the output shape of the previous layer
        """
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs],
                                      initializer=self.kernel_initializer)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """Perform the forward pass.

        Arguments:
            input {tf.Tensor} -- the previous layer's output

        Returns:
            tf.Tensor -- the result
        """
        return tf.matmul(input, self.kernel) + self.bias
