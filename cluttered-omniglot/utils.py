import numpy as np
import tensorflow as tf

from scipy import special
from tensorflow.python.framework import ops


def rotation_regularizer(scale, scope=None):
    """Performs custom regularizer.
    """
    def rotate(X):
        # perform rotation
        return tf.reverse(tf.transpose(X, perm=[1, 0, 2, 3]), axis=[0])

    def rr(weights):
        with ops.name_scope(scope, 'rotation_regularizer', [weights]) as name:
            my_scale = ops.convert_to_tensor(scale,
                                             dtype=weights.dtype.base_dtype,
                                             name='scale')
            dims = weights.get_shape().as_list()

            rotation_loss = tf.Variable(0.0)
            slices = []
            for i in range(0, 4):
                cur_slice = weights[:, :, :, (dims[-1] * i) // 8 : (dims[-1] * (i + 1)) // 8]
                slices.append(cur_slice)
            for i in range(0, 4):
                # ensure difference between next and current slices rotated is not too different
                cur_slice = slices[i]
                next_slice = slices[(i + 1) % 4]
                rotation_loss = rotation_loss + tf.abs(tf.reduce_sum(
                    cur_slice - rotate(next_slice)))
            # normalize
            rotation_loss /= tf.reduce_sum(tf.abs(weights))
            return rotation_loss * scale

    return rr

def sum_regularizer(regularizer_list, scope=None):
  """Returns a function that applies the sum of multiple regularizers.
  Args:
    regularizer_list: A list of regularizers to apply.
    scope: An optional scope name
  Returns:
    A function with signature `sum_reg(weights)` that applies the
    sum of all the input regularizers.
  """
  regularizer_list = [reg for reg in regularizer_list if reg is not None]
  if not regularizer_list:
    return None

  def sum_reg(weights):
    """Applies the sum of all the input regularizers."""
    with ops.name_scope(scope, 'sum_regularizer', [weights]) as name:
      regularizer_tensors = [reg(weights) for reg in regularizer_list]
      return math_ops.add_n(regularizer_tensors, name=name)

  return sum_reg


def softmax(x, axis=None):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x - special.logsumexp(x, axis=axis, keepdims=True))
