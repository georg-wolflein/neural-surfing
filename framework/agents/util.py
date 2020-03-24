import tensorflow as tf


def get_num_weights(model: tf.keras.Model) -> int:
    weights_shape = map(tf.shape, model.get_weights())
    return sum(map(tf.reduce_prod, weights_shape))
