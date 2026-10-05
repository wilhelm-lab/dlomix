"""Retention-time evaluation metrics, implemented once for both backends.

Written against ``keras.ops`` so the TensorFlow and PyTorch backends share a
single definition -- see :mod:`dlomix.losses.intensity` for the rationale.
``keras.metrics.Metric`` is itself backend-agnostic, so :class:`TimeDeltaMetric`
is usable both as a Keras metric and as a plain callable in a PyTorch loop.
"""

import keras
from keras import ops

# Parts of the code adopted and modified based on:
# https://github.com/horsepurve/DeepRTplus/blob/cde829ef4bd8b38a216d668cf79757c07133b34b/RTdata_emb.py


def _percentile_of_absolute_error(y_true, y_pred, percentage, normalize):
    """Nth percentile of the absolute error, optionally range-normalized."""
    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)

    # Note: Flatten both tensors before computing abs error.
    # Tensors with shape (batch, 1) -- common with a Dense(1) output -- otherwise
    # make sort operate row-wise instead of across all values, and the element
    # count captures only the batch dimension.
    y_true_flat = ops.reshape(y_true, [-1])
    y_pred_flat = ops.reshape(y_pred, [-1])

    abs_error = ops.abs(y_true_flat - y_pred_flat)

    n = ops.cast(ops.size(abs_error), "float32")
    percentile_index = ops.cast(n * percentage, "int32")

    delta = ops.take(ops.sort(abs_error), percentile_index - 1)

    if normalize:
        norm_range = ops.max(y_true_flat) - ops.min(y_true_flat)
        return delta / norm_range
    return delta


@keras.saving.register_keras_serializable(package="dlomix")
class TimeDeltaMetric(keras.metrics.Metric):
    """
    Implementation of the time delta metric as a Keras Metric using subclassing.

    Parameters
    ----------
    percentage : float, optional
        What percentage of the data points to consider. Defaults to 0.95.
    name : str, optional
        Name of the metric. Defaults to 'timedelta'.
    double_delta : bool, optional
        Whether to multiply the computed delta by 2 to make it two-sided. Defaults to False.
    normalize : bool, optional
        Whether to normalize the delta by the range of the true values. Defaults to False.

    Notes
    -----
    The reported value is the mean of per-batch percentiles, which is an approximation
    of the true dataset-level percentile. This is a known trade-off in streaming metrics.
    For an exact result, compute offline with numpy over the full dataset.

    This class replaces the previous PyTorch-only ``TimeDeltaMetric(percentage,
    normalize)`` callable. It keeps the stateful Keras API and accepts ``normalize``
    as a keyword, and it can still be called directly for one-shot evaluation.
    """

    def __init__(
        self,
        percentage=0.95,
        name="timedelta",
        double_delta=False,
        normalize=False,
        **kwargs,
    ):
        super(TimeDeltaMetric, self).__init__(name=name, **kwargs)
        self.delta = self.add_weight(name="delta", initializer="zeros")
        self.batch_count = self.add_weight(name="batch-count", initializer="zeros")
        self.percentage = percentage
        self.double_delta = double_delta
        self.normalize = normalize

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        delta_value = _percentile_of_absolute_error(
            y_true, y_pred, self.percentage, self.normalize
        )

        if self.double_delta:
            delta_value = delta_value * 2

        self.batch_count.assign_add(1.0)
        self.delta.assign_add(ops.cast(delta_value, self.delta.dtype))

    def result(self):
        return self.delta / self.batch_count

    def reset_state(self):
        self.delta.assign(0.0)
        self.batch_count.assign(0.0)

    def __call__(self, y_true, y_pred, **kwargs):
        """Evaluate in one shot, without touching the streaming state.

        Keeps the call-and-get-a-number usage the PyTorch implementation offered.
        """
        delta_value = _percentile_of_absolute_error(
            y_true, y_pred, self.percentage, self.normalize
        )
        return delta_value * 2 if self.double_delta else delta_value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "percentage": self.percentage,
                "double_delta": self.double_delta,
                "normalize": self.normalize,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="dlomix")
def timedelta(y_true, y_pred, normalize=False, percentage=0.95):
    """
    Functional implementation of the time delta metric.

    Computes the Nth percentile of the absolute error between true and predicted values.

    Parameters
    ----------
    y_true : tensor
        True values of the target.
    y_pred : tensor
        Predicted values of the target.
    normalize : bool, optional
        Whether to normalize the delta by the range of the true values. Defaults to False.
    percentage : float, optional
        Percentile threshold. Defaults to 0.95.

    Returns
    -------
    tensor
        The Nth percentile of the absolute error, as a scalar.

    Notes
    -----
    The PyTorch-only version of this function took ``(percentage, normalize)`` as
    its third and fourth arguments. This unified version follows the TensorFlow
    order, ``(normalize, percentage)``; pass them by keyword to be unambiguous.
    """
    return _percentile_of_absolute_error(y_true, y_pred, percentage, normalize)
