import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp


class TimeDeltaMetric(tf.keras.metrics.Metric):
    """
    Implementation of the time delta metric as a Keras Metric.

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

    def __init__(
        self,
        mean=0,
        std=1,
        percentage=0.95,
        name="timedelta",
        rescale_targets=False,
        rescale_predictions=False,
        double_delta=False,
        **kwargs
    ):
        super(TimeDeltaMetric, self).__init__(name=name, **kwargs)
        self.delta = self.add_weight(name="delta", initializer="zeros")
        self.batch_count = self.add_weight(name="batch-count", initializer="zeros")
        self.mean = mean
        self.std = std
        self.percentage = percentage
        self.rescale_targets = rescale_targets
        self.rescale_predictions = rescale_predictions
        self.double_delta = double_delta

    def update_state(self, y_true, y_pred, sample_weight=None):
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

        # rescale
        if self.rescale_targets:
            y_true = y_true * self.std + self.mean

        if self.rescale_predictions:
            y_pred = y_pred * self.std + self.mean

        # find position of the index
        length = tf.shape(y_true)[0]
        mark = tf.cast(length, dtype=tf.float32) * self.percentage
        mark = tf.cast(mark, dtype=tf.int32)

        # compute residuals and sort
        abs_error = tf.abs(y_true - y_pred)
        d = tf.sort(abs_error)[mark - 1]

        # two-sided delta
        if self.double_delta:
            d = d * 2

        # update count of batches
        self.batch_count.assign_add(1.0)

        # update delta
        self.delta.assign_add(tf.math.reduce_sum(d))

    def result(self):
        # this is simple averaging over the batches, more complex reduction can be added based on domain expertise
        # Examples are: take max or min of both deltas (translates to a strict or a relaxed metric)
        return tf.math.divide(self.delta, self.batch_count)


# code adopted and modified based on:
# https://github.com/horsepurve/DeepRTplus/blob/cde829ef4bd8b38a216d668cf79757c07133b34b/RTdata_emb.py
def delta95_metric(y_true, y_pred):
    mark95 = tf.cast(
        tf.cast(tf.shape(y_true)[0], dtype=tf.float32) * 0.95, dtype=tf.int32
    )
    abs_error = K.abs(y_true - y_pred)
    delta = tf.sort(abs_error)[mark95 - 1]
    norm_range = K.max(y_true) - K.min(y_true)
    return (delta * 2) / (norm_range)


def TimeDeltaMetric2():
    """
    The 95th percentile of absolute error between label and prediction

    """

    def calc_metric(y_true, y_pred, sample_weight=None):
        # compute residuals and sort
        abs_error = tf.abs(y_true - y_pred)
        return tfp.stats.percentile(abs_error, 95)

    calc_metric.__name__ = "delta95"
    return calc_metric


METRICS_DICT = {
    "delta95": TimeDeltaMetric(),
}
