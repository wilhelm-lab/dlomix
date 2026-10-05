"""Charge-state evaluation metrics, implemented once for both backends.

Written against ``keras.ops`` so the TensorFlow and PyTorch backends share a
single definition -- see :mod:`dlomix.losses.intensity` for the rationale.
"""

import keras
from keras import ops


def _pairwise_mask(y_true, y_pred):
    """Mask keeping every component that is non-zero in at least one vector.

    Only components that are zero in *both* vectors are discarded. The previous
    PyTorch implementation required both to be non-zero, which discarded genuine
    prediction errors and made the two backends report different numbers; this
    is the TensorFlow behaviour, and the one the docstrings describe.
    """
    both_zero = ops.logical_and(ops.equal(y_true, 0.0), ops.equal(y_pred, 0.0))
    return ops.cast(ops.logical_not(both_zero), "float32")


def _adjusted_error(y_true, y_pred, error_fn):
    y_true = ops.cast(ops.convert_to_tensor(y_true), "float32")
    y_pred = ops.cast(ops.convert_to_tensor(y_pred), "float32")

    mask = _pairwise_mask(y_true, y_pred)

    errors = error_fn(y_true * mask - y_pred * mask)
    count_non_zero = ops.sum(mask)

    # Avoid division by zero by adding a small epsilon to the denominator
    return ops.sum(errors) / (count_non_zero + keras.config.epsilon())


@keras.saving.register_keras_serializable(package="dlomix")
def adjusted_mean_absolute_error(y_true, y_pred):
    """
    Used as an evaluation metric for charge state prediction.

    For two vectors, discard those components that
    are 0 in both vectors and compute the mean
    absolute error for the adjusted vector.

    Parameters
    ----------
    y_true : tensor
        Ground-truth charge state vector.
    y_pred : tensor
        Predicted charge state vector, with the same shape as `y_true`.

    Returns
    -------
    tensor
        A scalar tensor with the adjusted mean absolute error.
    """
    return _adjusted_error(y_true, y_pred, ops.abs)


@keras.saving.register_keras_serializable(package="dlomix")
def adjusted_mean_squared_error(y_true, y_pred):
    """
    For two vectors, discard those components that
    are 0 in both vectors and compute the mean
    squared error for the adjusted vector.

    Parameters
    ----------
    y_true : tensor
        Ground-truth charge state vector.
    y_pred : tensor
        Predicted charge state vector, with the same shape as `y_true`.

    Returns
    -------
    tensor
        A scalar tensor with the adjusted mean squared error.
    """
    return _adjusted_error(y_true, y_pred, ops.square)
