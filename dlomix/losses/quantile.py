import tensorflow as tf

class QuantileLoss(tf.keras.losses.Loss):
    '''
    Quantile loss (pinball loss) 
    '''
    def __init__(self, quantile=tf.constant([[0.1, 0.9]]), name="quantile_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantile = quantile

    def call(self, y_true, y_pred):
        err = tf.subtract(y_true, y_pred)
        return tf.reduce_sum(tf.maximum(self.quantile * err, (self.quantile - 1) * err), axis=-1)

    def get_config(self):
        config = {
            'quantile': self.quantile
        }
        base_config = super().get_config()
        return {**base_config, **config}