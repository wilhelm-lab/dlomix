import tensorflow as tf

from dlomix.eval import tf as tf_eval
from dlomix.eval import torch as torch_eval
from dlomix.types import Tensor


def timedelta(y_true: Tensor, y_pred: Tensor, **kwargs) -> Tensor:
    """Find value that is below 95th percentile of absolute error.
    Scale this by the range of the true values (max-min)

    Parameters
    ----------
    y_true : Tensor
        ground truth
    y_pred : Tensor
        predictions

    Returns
    -------
    Tensor
        95% percentile of absolute error squared divided by range of y_true
    """
    if isinstance(y_true, tf.Tensor):
        ret = tf_eval.rt_eval.timedelta(y_true, y_pred, **kwargs)
    else:
        ret = torch_eval.rt_eval.timedelta(y_true, y_pred, **kwargs)
    return ret
