import tensorflow as tf
import tensorflow.keras.backend as K


def delta95_metric(y_true, y_pred):
    mark95 = tf.cast(
        tf.cast(tf.shape(y_true)[0], dtype=tf.float32) * 0.95, dtype=tf.int32)
    abs_error = K.abs(y_true - y_pred)
    delta = tf.sort(abs_error)[mark95 - 1]
    norm_range = K.max(y_true) - K.min(y_true)
    return (delta * 2) / (norm_range)


def delta99_metric(y_true, y_pred):
    mark99 = tf.cast(
        tf.cast(tf.shape(y_true)[0], dtype=tf.float64) * 0.99, dtype=tf.int32)
    abs_error = tf.abs(y_true - y_pred)
    delta = tf.sort(abs_error)[mark99 - 1]
    norm_range = K.max(y_true) - K.min(y_true)
    return (delta * 2) / (norm_range)


class TimeDeltaMetric(tf.keras.metrics.Metric):
    def __init__(self, mean=0, std=1, percentage=0.95, name='timedelta', rescale_targets=True, **kwargs):
        super(TimeDeltaMetric, self).__init__(name=name, **kwargs)
        self.delta = self.add_weight(name='delta', initializer='zeros')
        self.mean = mean
        self.std = std
        self.percentage = percentage
        self.rescale_targets = rescale_targets

    def update_state(self, y_true, y_pred, sample_weight=None):

        # rescale
        if self.rescale_targets:
            y_true = y_true * self.std + self.mean

        y_pred = y_pred * self.std + self.mean

        # find position of the index
        length = tf.shape(y_true)[0]
        mark = tf.cast(length, dtype=tf.float32) * self.percentage
        mark = tf.cast(mark, dtype=tf.int32)

        # compute residuals and sort
        abs_error = tf.abs(y_true - y_pred)
        delta = tf.sort(abs_error)[mark - 1]

        # two-sided delta
        self.delta.assign(tf.reduce_sum(delta * 2))

    def result(self):
        return self.delta

    def reset_state(self):
        self.delta.assign(0.)
