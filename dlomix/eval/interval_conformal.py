import tensorflow as tf

class IntervalSize(tf.keras.losses.Loss):
    '''
    Size of the prediction interval.
    '''
    def __init__(self, name="interval_size", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        interval_size = tf.subtract(y_pred[:,1], y_pred[:,0])
        return interval_size

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}


class AbsoluteIntervalSize(tf.keras.losses.Loss):
    '''
    Absolute size of the prediction interval.
    '''
    def __init__(self, name="abs_interval_size", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        abs_interval_size = tf.math.abs(tf.subtract(y_pred[:,1], y_pred[:,0]))
        return abs_interval_size

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}


class RelativeCentralDistance(tf.keras.losses.Loss):
    '''
    Distance of the true value from the center of the prediction intverval,
    divided by half of the the prediction interval size.
    The result is 0 if true value is at center of the prediction interval, 
    1 if at (symm.) interval boundary and >1 else.
    '''
    def __init__(self, name="relative_central_distance", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        # absolute distance between the two interval boundaries 
        # (if upper boudary is below lower, this is nevertheless positive)
        interval_size = tf.math.abs(tf.subtract(y_pred[:,1], y_pred[:,0]))
        # absolute distance of the true value from the inverval center/mean
        central_dist = tf.math.abs(tf.subtract(y_true[:,0], tf.math.reduce_mean(y_pred, axis=-1)))
        # divide by half of interval size
        res = tf.divide(central_dist, tf.divide(interval_size, 2.))
        indices = tf.where(tf.math.is_inf(res))
        return tf.tensor_scatter_nd_update(
            res,
            indices,
            tf.ones((tf.shape(indices)[0])) * 0.
        )

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}


class IntervalConformalScore(tf.keras.losses.Loss):
    '''
    Computes conformal scores for prediction intervals pred_intervals and true values y_true.
    '''
    def __init__(self, name="interval_conformal_score", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, pred_intervals):
        return tf.reduce_max(tf.stack([tf.subtract(pred_intervals, y_true)[:,0], -tf.subtract(pred_intervals, y_true)[:,1]], 1), -1)

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}


class IntervalConformalQuantile(tf.keras.losses.Loss):
    '''
    Computes the conformal quantile based on the distribution of conformal scores
    for prediction intervals pred_intervals and true values y_true.
    '''
    def __init__(self, alpha=0.1, name="interval_conformal_quantile", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha

    def call(self, y_true, pred_intervals):
        scores = tf.reduce_max(tf.stack([tf.subtract(pred_intervals, y_true)[:,0], -tf.subtract(pred_intervals, y_true)[:,1]], 1), -1)
        n = tf.cast(tf.shape(y_true)[0], tf.float32) #without casting to float, next line throws an error
        q = tf.math.ceil((n + 1.) * (1. - self.alpha)) / n
        tfp_quantile = tf.sort(scores, axis=-1, direction='ASCENDING', name=None)[int(q * n)]
        return tfp_quantile

    def get_config(self):
        config = {'alpha' : self.alpha}
        base_config = super().get_config()
        return {**base_config, **config}