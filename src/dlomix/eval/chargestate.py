import tensorflow as tf

from dlomix.eval import tf as tf_eval
from dlomix.eval import torch as torch_eval
from dlomix.types import Tensor


def adjusted_mean_absolute_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Used as an evaluation metric for charge state prediction.

    For two vectors, discard those components that
    are 0 in both vectors and compute the mean
    absolute error for the adjusted vector.
    """
    if isinstance(y_true, tf.Tensor):
        ret = tf_eval.chargestate.adjusted_mean_absolute_error(y_true, y_pred)
    else:
        ret = torch_eval.chargestate.adjusted_mean_absolute_error(y_true, y_pred)
    return ret


def adjusted_mean_squared_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    For two vectors, discard those components that
    are 0 in both vectors and compute the mean
    squared error for the adjusted vector.
    """
    if isinstance(y_true, tf.Tensor):
        ret = tf_eval.chargestate.adjusted_mean_squared_error(y_true, y_pred)
    else:
        ret = torch_eval.chargestate.adjusted_mean_squared_error(y_true, y_pred)
    return ret
