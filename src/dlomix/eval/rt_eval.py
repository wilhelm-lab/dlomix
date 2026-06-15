import tensorflow as tf
import tensorflow.keras.backend as K

# Parts of the code adopted and modified based on:
# https://github.com/horsepurve/DeepRTplus/blob/cde829ef4bd8b38a216d668cf79757c07133b34b/RTdata_emb.py


@tf.keras.utils.register_keras_serializable(package="dlomix")
class TimeDeltaMetric(tf.keras.metrics.Metric):
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

    Notes
    -----
    The reported value is the mean of per-batch percentiles, which is an approximation
    of the true dataset-level percentile. This is a known trade-off in streaming metrics.
    For an exact result, compute offline with numpy over the full dataset.
    """

    def __init__(self, percentage=0.95, name="timedelta", double_delta=False, **kwargs):
        super(TimeDeltaMetric, self).__init__(name=name, **kwargs)
        self.delta = self.add_weight(name="delta", initializer="zeros")
        self.batch_count = self.add_weight(name="batch-count", initializer="zeros")
        self.percentage = percentage
        self.double_delta = double_delta

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None
    ) -> None:
        # Note: Flatten both tensors before computing abs error.
        # Previously, tensors with shape (batch, 1) — common with Dense(1) output —
        # caused tf.sort to operate row-wise instead of across all values, and
        # tf.shape(...)[0] only captured the batch dimension, not total elements.
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        abs_error = tf.abs(y_true_flat - y_pred_flat)

        sorted_error = tf.sort(abs_error)

        n = tf.cast(tf.size(sorted_error), tf.float32)
        percentile_index = tf.cast(n * self.percentage, dtype=tf.int32)

        delta_value = sorted_error[percentile_index - 1]

        if self.double_delta:
            delta_value = delta_value * 2

        self.batch_count.assign_add(1.0)
        self.delta.assign_add(delta_value)

    def result(self):
        return self.delta / self.batch_count

    def reset_states(self):
        self.delta.assign(0.0)
        self.batch_count.assign(0.0)

    def get_config(self):
        return {
            "percentage": self.percentage,
            "double_delta": self.double_delta,
            "name": self.name,
        }


@tf.keras.utils.register_keras_serializable("dlomix")
def timedelta(y_true, y_pred, normalize=False, percentage=0.95):
    """
    Functional implementation of the time delta metric.

    Computes the Nth percentile of the absolute error between true and predicted values.

    Parameters
    ----------
    y_true : tf.Tensor
        True values of the target.
    y_pred : tf.Tensor
        Predicted values of the target.
    normalize : bool, optional
        Whether to normalize the delta by the range of the true values. Defaults to False.
    percentage : float, optional
        Percentile threshold. Defaults to 0.95.

    Returns
    -------
    tf.Tensor
        The Nth percentile of the absolute error, as a scalar.
    """
    # Note: Flatten both inputs before computing abs error (see TimeDeltaMetric note).
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    abs_error = K.abs(y_true_flat - y_pred_flat)

    n = tf.cast(tf.size(abs_error), dtype=tf.float32)
    mark_percentile = tf.cast(n * percentage, dtype=tf.int32)

    delta = tf.sort(abs_error)[mark_percentile - 1]

    if normalize:
        norm_range = K.max(y_true_flat) - K.min(y_true_flat)
        return delta / norm_range
    return delta
