import tensorflow as tf


class DenseWithFixedBias(tf.keras.layers.Layer):
    def __init__(self, num_outputs: int, bias: float, kernel: tf.Tensor):
        super().__init__()
        self.num_outputs = num_outputs
        self.bias = bias
        self.kernel_initializer = tf.constant_initializer(kernel)

    def build(self, input_shape: tf.TensorShape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs],
                                      initializer=self.kernel_initializer)

    def call(self, input):
        return tf.matmul(input, self.kernel) + self.bias
