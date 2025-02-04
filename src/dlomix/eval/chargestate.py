import tensorflow as tf
from keras import backend as K

from dlomix.eval import tf as tf_eval


def adjusted_mean_absolute_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Used as an evaluation metric for charge state prediction.

    For two vectors, discard those components that
    are 0 in both vectors and compute the mean
    absolute error for the adjusted vector.
    """
    if isinstance(y_true, tf.Tensor):
        ret = tf_eval.chargestate.adjusted_mean_absolute_error(y_true, y_pred)
        return ret
    else:
        raise NotImplementedError("todo")


def adjusted_mean_squared_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    For two vectors, discard those components that
    are 0 in both vectors and compute the mean
    squared error for the adjusted vector.
    """
    if isinstance(y_true, tf.Tensor):
        ret = tf_eval.chargestate.adjusted_mean_squared_error(y_true, y_pred)
        return ret
    else:
        raise NotImplementedError("todo")


if __name__ == "__main__":
    import numpy as np

    y_true = K.constant([0, 1, 2, 2, 0, 0, 0, 0], dtype="float32")
    y_pred = K.constant([0, 3, 0, 4, 0, 0, 2, 0], dtype="float32")
    y_true, y_pred = K.to_dense(y_true), K.to_dense(y_pred)
    mae = adjusted_mean_absolute_error(y_true, y_pred)
    mse = adjusted_mean_squared_error(y_true, y_pred)
    assert np.isclose(mae, 2.0)
    assert np.isclose(mse, 4.0)
    print(f"Adjusted MAE: {mae:.4f}")
    print(f"Adjusted MSE: {mse:.4f}")
