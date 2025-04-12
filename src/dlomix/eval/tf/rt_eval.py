import tensorflow as tf
import tensorflow.keras.backend as K

# Parts of the code adopted and modified based on:
# https://github.com/horsepurve/DeepRTplus/blob/cde829ef4bd8b38a216d668cf79757c07133b34b/RTdata_emb.py


class TimeDeltaMetric(tf.keras.metrics.Metric):
    """
    Implementation of the time delta metric as a Keras Metric using subclassing.

    Parameters
    ----------
    mean : int, optional
        Mean value of the targets in case normalization was performed. Defaults to 0.
    std : int, optional
        Standard deviation value of the targets in case normalization was performed. Defaults to 1.
    percentage : float, optional
        What percentage of the data points to consider, this is specific to the computation of the metric. Defaults to 0.95 which corresponds to 95% of the data points and is the mostly used value in papers.
    name : str, optional
        Name of the metric so that it can be reported and used later in Keras History objects. Defaults to 'timedelta'.
    rescale_targets : bool, optional
        Whether to rescale (denormalize) targets or not. Defaults to False.
    rescale_predictions : bool, optional
        Whether to rescale (denormalize) predictions or not. Defaults to False.
    double_delta : bool, optional
        Whether to multiply the computed delta by 2 in order to make it two-sided or not. Defaults to False.
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
        """
        Update the metric state.

        Parameters
        ----------
        y_true : tf.Tensor
            True values of the target.
        y_pred : tf.Tensor
            Predicted values of the target.
        sample_weight : tf.Tensor, optional
            Sample weights. Defaults to None.
        """

        # Compute absolute errors
        abs_error = tf.abs(y_true - y_pred)

        # Sort the errors in ascending order
        sorted_error = tf.sort(abs_error)

        # Find the index corresponding to the 95th percentile
        # First, cast the shape to float, then multiply by 0.95, and finally cast back to int32 for indexing
        percentile_index = tf.cast(
            tf.cast(tf.shape(sorted_error)[0], tf.float32) * self.percentage,
            dtype=tf.int32,
        )

        # Select the 95th percentile value
        delta_value = sorted_error[
            percentile_index - 1
        ]  # tf.shape gives a 0-based index

        # two-sided delta
        if self.double_delta:
            delta_value = delta_value * 2

        # Update the count of batches
        self.batch_count.assign_add(1.0)

        # Update delta (to track sum of deltas over batches)
        self.delta.assign_add(delta_value)

    def result(self):
        # Return the average of deltas over batches
        return self.delta / self.batch_count

    def reset_states(self):
        # Reset the state at the beginning of each epoch
        self.delta.assign(0.0)
        self.batch_count.assign(0.0)


def timedelta(y_true, y_pred, normalize=False, percentage=0.95):
    """
    A functional implementation to compute the 95th percentile of the absolute error between true and predicted values.
    Parameters
    ----------
    y_true : tf.Tensor
        True values of the target.
    y_pred : tf.Tensor
        Predicted values of the target.
    normalize : bool, optional
        Whether to normalize the delta by the range of the true values. Defaults to False.
    percentage : float, optional
        What percentage of the data points to consider, this is specific to the computation of the metric. Defaults to 0.95 which corresponds to 95% of the data points and is the mostly used value in papers.
    Returns
    -------
    tf.Tensor
        The 95th percentile of the absolute error.
    """

    mark_percentile = tf.cast(
        tf.cast(tf.shape(y_true)[0], dtype=tf.float32) * percentage, dtype=tf.int32
    )

    abs_error = K.abs(y_true - y_pred)
    delta = tf.sort(abs_error)[mark_percentile - 1]

    if normalize:
        norm_range = K.max(y_true) - K.min(y_true)
        return (delta) / (norm_range)
    return delta
