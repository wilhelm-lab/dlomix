import tensorflow as tf


class MetaEncoder(tf.keras.layers.Layer):
    def __init__(self, regressor_layer_size=512, dropout_rate=0.2):
        super(MetaEncoder, self).__init__()
        self.concat = tf.keras.layers.Concatenate(name="meta_in")
        self.dense = tf.keras.layers.Dense(regressor_layer_size, name="meta_dense")
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name="meta_dense_do")

    def call(self, inputs, **kwargs):
        x = self.concat(inputs)
        x = self.dense(x)
        x = self.dropout(x)

        return x
