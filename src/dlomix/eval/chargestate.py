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


if __name__ == "__main__":
    import os

    os.chdir("./..")
    import numpy as np
    import torch

    y_true = [0, 1, 2, 2, 0, 0, 0, 0]
    y_pred = [0, 3, 0, 4, 0, 0, 2, 0]

    # y_true = K.constant(y_true, dtype="float32")
    # y_pred = K.constant(y_pred, dtype="float32")

    y_true = torch.tensor(data=y_true, dtype=torch.float32)
    y_pred = torch.tensor([y_pred], dtype=torch.float32)

    mae = adjusted_mean_absolute_error(y_true, y_pred)
    mse = adjusted_mean_squared_error(y_true, y_pred)
    assert np.isclose(mae, 2.0)
    assert np.isclose(mse, 4.0)
    print(f"Adjusted MAE: {mae:.4f}")
    print(f"Adjusted MSE: {mse:.4f}")
