import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from dlpro.constants import ALPHABET_UNMOD
from dlpro.layers.attention import AttentionLayer


class PrositRetentionTimePredictor(tf.keras.Model):
    def __init__(self, embedding_input_dim=23, embedding_output_dim=16, seq_length=30, vocab_dict=ALPHABET_UNMOD,
                 encoder_layer_type="gru"):
        super(PrositRetentionTimePredictor, self).__init__()

        self.string_lookup = preprocessing.StringLookup(vocabulary=list(vocab_dict.keys()))

        self.embedding = tf.keras.layers.Embedding(input_dim=embedding_input_dim,
                                                   output_dim=embedding_output_dim,
                                                   input_length=seq_length)
        self._build_encoder()

        self.attention = AttentionLayer()

        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(rate=0.1)
        ])

        self.output_layer = tf.keras.layers.Dense(1)

    def _build_encoder(self):
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=256, return_sequences=True)),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.GRU(units=512, return_sequences=True),
            tf.keras.layers.Dropout(rate=0.5)
        ])

    def call(self, inputs, **kwargs):
        x = self.string_lookup(inputs)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        return x
