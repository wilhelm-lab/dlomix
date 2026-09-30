"""Intensity losses, implemented once for both backends.

These are pure elementwise tensor math with no layer state, so they are written
against ``keras.ops`` and run unchanged on the TensorFlow and PyTorch backends.
:mod:`dlomix.config` aligns ``KERAS_BACKEND`` with ``DLOMIX_BACKEND`` at import
time, so the active framework follows the one the user selected.

They accept and return that backend's native tensors, and remain differentiable
under both ``tf.GradientTape`` and ``torch.autograd``.
"""

import keras
import numpy as np
from keras import ops

# tf.nn.l2_normalize floors the squared sum at this value rather than adding an
# epsilon to the norm. Replicated exactly so results stay comparable with models
# trained before the backends were unified.
_L2_NORMALIZE_EPSILON = 1e-12


def _l2_normalize(x, axis=-1):
    """Backend-agnostic equivalent of ``tf.nn.l2_normalize``."""
    square_sum = ops.sum(ops.square(x), axis=axis, keepdims=True)
    return x / ops.sqrt(ops.maximum(square_sum, _L2_NORMALIZE_EPSILON))


def _mask_negative_peaks(y_true, y_pred):
    """Zero out the peaks that cannot be there (encoded as -1).

    Multiplying by ``(y_true + 1)`` cancels those positions, since the padded
    entries carry a value of -1.
    """
    epsilon = keras.config.epsilon()
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)
    return true_masked, pred_masked


@keras.saving.register_keras_serializable(package="dlomix")
def masked_spectral_distance(y_true, y_pred):
    """
    Calculates the masked spectral distance between true and predicted intensity vectors.
    The masked spectral distance is a metric for comparing the similarity between two intensity vectors.

    Masked, normalized spectral angles between true and pred vectors

    > arccos(1*1 + 0*0) = 0 -> SL = 0 -> high correlation

    > arccos(0*1 + 1*0) = pi/2 -> SL = 1 -> low correlation

    Parameters
    ----------
    y_true : tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_pred : tensor
        A tensor containing the predicted values, with the same shape as `y_true`.

    Returns
    -------
    tensor
        A tensor of per-sample spectral distances, with shape `(batch_size,)`.

    Notes
    -----
    The result is **per sample**, not reduced. Keras reduces it automatically when
    used as a loss; a hand-written PyTorch training loop must call ``.mean()``
    (or another reduction) before ``.backward()``.
    """
    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)

    true_masked, pred_masked = _mask_negative_peaks(y_true, y_pred)

    pred_norm = _l2_normalize(pred_masked, axis=-1)
    true_norm = _l2_normalize(true_masked, axis=-1)

    # Spectral Angle (SA) calculation
    # (from the definition below, it is clear that ions with higher intensities
    #  will always have a higher contribution)
    product = ops.sum(pred_norm * true_norm, axis=-1)
    # Rounding error can push the dot product of two unit vectors just outside
    # [-1, 1], where arccos returns NaN. Clipping keeps a perfect match at 0.
    product = ops.clip(product, -1.0, 1.0)
    arccos = ops.arccos(product)
    return 2 * arccos / np.pi


@keras.saving.register_keras_serializable(package="dlomix")
def masked_pearson_correlation_distance(y_true, y_pred):
    """
    Calculates the masked Pearson correlation distance between true and predicted intensity vectors.
    The masked Pearson correlation distance is a metric for comparing the similarity between two intensity vectors,
    taking into account only the non-negative values in the true values tensor (which represent valid peaks).

    Parameters
    ----------
    y_true : tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_pred : tensor
        A tensor containing the predicted values, with the same shape as `y_true`.

    Returns
    -------
    tensor
        A scalar tensor containing the masked Pearson correlation distance,
        computed over all elements of the batch.
    """
    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)

    true_masked, pred_masked = _mask_negative_peaks(y_true, y_pred)

    # Reduced over every axis, so a single scalar is returned.
    mx = ops.mean(true_masked)
    my = ops.mean(pred_masked)
    xm, ym = true_masked - mx, pred_masked - my
    r_num = ops.mean(xm * ym)
    # Population standard deviation, matching tf.math.reduce_std and
    # torch.std(unbiased=False).
    r_den = ops.std(xm) * ops.std(ym)
    return 1 - (r_num / r_den)
