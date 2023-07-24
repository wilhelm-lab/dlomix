import tensorflow as tf

class ScalarConformalScore(tf.keras.losses.Loss):
    '''
    Computes conformal scores for predicted scalar error estimates pred_err and true values y_true.
    '''
    def __init__(self, name="scalar_conformal_scores", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, pred_err):
        scores = tf.divide(tf.math.abs(y_true - pred_err[:,0]), pred_err[:,1])
        return scores
    
    def get_config(self):
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}

class ScalarConformalQuantile(tf.keras.losses.Loss):
    '''
    Computes the conformal quantile based on the distribution of conformal scores
    for scalar error estimates pred_err and true values y_true.
    '''
    def __init__(self, alpha=0.1, name="scalar_conformal_quantile", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha

    def call(self, y_true, pred_err):
        scores = tf.divide(tf.math.abs(y_true - pred_err[:,0]), pred_err[:,1])
        n = tf.cast(tf.shape(y_true)[0], tf.float32) #without casting to float, next line throws an error
        q = tf.math.ceil((n + 1.)*(1. - self.alpha)) / n
        tfp_quantile = tf.sort(scores, axis=-1, direction='ASCENDING', name=None)[int(q * n)]
        return tfp_quantile

    def get_config(self):
        config = {'alpha' : self.alpha}
        base_config = super().get_config()
        return {**base_config, **config}