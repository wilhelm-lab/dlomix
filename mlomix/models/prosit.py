import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from mlomix.constants import ALPHABET_UNMOD
from mlomix.layers.attention import AttentionLayer


class PrositRetentionTimePredictor(tf.keras.Model):
    def __init__(self, embedding_output_dim=16, seq_length=30, vocab_dict=ALPHABET_UNMOD,
                 dropout_rate=0.5, latent_dropout_rate=0.1, recurrent_layers_sizes=(256, 512), regressor_layer_size=512):
        super(PrositRetentionTimePredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(vocab_dict) + 2

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes

        self.string_lookup = preprocessing.StringLookup(vocabulary=list(vocab_dict.keys()))

        self.embedding = tf.keras.layers.Embedding(input_dim=self.embeddings_count,
                                                   output_dim=embedding_output_dim,
                                                   input_length=seq_length)
        self._build_encoder()

        self.attention = AttentionLayer()

        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Dense(self.regressor_layer_size, activation='relu'),
            tf.keras.layers.Dropout(rate=self.latent_dropout_rate)
        ])

        self.output_layer = tf.keras.layers.Dense(1)

    def _build_encoder(self):
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=self.recurrent_layers_sizes[0], return_sequences=True)),
            tf.keras.layers.Dropout(rate=self.dropout_rate),
            tf.keras.layers.GRU(units=self.recurrent_layers_sizes[1], return_sequences=True),
            tf.keras.layers.Dropout(rate=self.dropout_rate)
        ])

    def call(self, inputs, **kwargs):
        x = self.string_lookup(inputs)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        return x
