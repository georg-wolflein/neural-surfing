import tensorflow as tf


def get_num_weights(model: tf.keras.Model) -> int:
    """Utility function to determine the number of weights in a keras model.

    Arguments:
        model {tf.keras.Model} -- the keras model

    Returns:
        int -- the number of weights
    """

    weights_shape = map(tf.shape, model.get_weights())
    return sum(map(tf.reduce_prod, weights_shape))


def get_distance_to_line(point: tf.Tensor, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Utility function to get the shortest distance from a point to the line passing through a and b.

    Arguments:
        point {tf.Tensor} -- the point
        a {tf.Tensor} -- one point on the line
        b {tf.Tensor} -- another point on the line

    Returns:
        tf.Tensor -- the distance
    """

    pa = point - a
    ba = b - a
    t = tf.tensordot(pa, ba, axes=[[0, 1], [1, 0]]) / \
        tf.tensordot(ba, ba, axes=[[0, 1], [1, 0]])
    d = tf.norm(pa - t * ba)
    return d


def scale_to_length(x: tf.Tensor, length: float) -> tf.Tensor:
    """Utility function to scale vector(s) to a certain length.

    Arguments:
        x {tf.Tensor} -- the vector (or collection of vectors in form of a matrix)
        length {float} -- the length

    Returns:
        tf.Tensor -- the scaled vector(s)
    """

    magnitude = tf.norm(x, axis=-1, keepdims=True)
    return (length / magnitude) * x
