import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from dlpro.constants import ALPHABET_UNMOD


class RetentionTimePredictor(tf.keras.Model):

    def __init__(self, embeddings_count, embedding_dim, seq_length=30, encoder="lstm", vocab_dict=ALPHABET_UNMOD):
        super(RetentionTimePredictor, self).__init__()

        self.string_lookup = preprocessing.StringLookup(vocabulary=list(vocab_dict.keys()))

        self.embedding = tf.keras.layers.Embedding(input_dim=embeddings_count + 1, output_dim=embedding_dim,
                                                   input_length=seq_length)

        self._build_encoder(encoder)

        self.flatten = tf.keras.layers.Flatten()
        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu')
        ])

        self.output_layer = tf.keras.layers.Dense(1)

    def _build_encoder(self, encoder_type):
        if encoder_type.lower() == 'conv1d':
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
                tf.keras.layers.Conv1D(filters=512, kernel_size=3, padding='valid', activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2)])
        else:
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.LSTM(256, return_sequences=True),
                tf.keras.layers.LSTM(256)
            ])

    def call(self, inputs, **kwargs):
        x = self.string_lookup(inputs)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.regressor(x)
        x = self.output_layer(x)

        return x

