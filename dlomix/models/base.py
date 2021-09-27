import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from dlomix.constants import ALPHABET_UNMOD


class RetentionTimePredictor(tf.keras.Model):

    def __init__(self, embedding_dim=16, seq_length=30,
                 encoder="conv1d", vocab_dict=ALPHABET_UNMOD):
        super(RetentionTimePredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(vocab_dict) + 2

        self.string_lookup = preprocessing.StringLookup(vocabulary=list(vocab_dict.keys()))

        self.embedding = tf.keras.layers.Embedding(input_dim=self.embeddings_count,
                                                   output_dim=embedding_dim,
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
                tf.keras.layers.LSTM(512)
            ])

    def call(self, inputs, **kwargs):
        x = self.string_lookup(inputs)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.regressor(x)
        x = self.output_layer(x)

        return x

