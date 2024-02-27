import tensorflow as tf
from ..constants import ALPHABET_UNMOD

'''
----------------------------------------------
TEST/EXPERIMENTAL MODELS
----------------------------------------------
'''


class LSTMTest(tf.keras.Model):
    '''
    Predicts the dominant charge state incorporating bidirectional LSTM layers.
    '''

    def __init__(self, embedding_dim=16, seq_length=30, vocab_dict=ALPHABET_UNMOD, num_classes=7, lstm_units=64):
        super(LSTMTest, self).__init__()
        # Model parameters
        self.embedding_dim = embedding_dim
        self.embeddings_count = len(vocab_dict) + 2
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.lstm_units = lstm_units

        # Model layers
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count, output_dim=self.embedding_dim, input_length=self.seq_length)
        self.bidirectional_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.lstm_units, return_sequences=True))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(self.seq_length, activation="relu")
        self.dense2 = tf.keras.layers.Dense(
            self.num_classes, activation="softmax")

    def call(self, inputs):
        # Forward pass
        x = self.embedding(inputs)
        x = self.bidirectional_lstm(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class LayerTestChargePredictor(tf.keras.Model):
    def __init__(self, seq_length=30, vocab_dict=ALPHABET_UNMOD, num_classes=7, lstm_units=64):
        super(LayerTestChargePredictor, self).__init__()
        # Model parameters
        self.seq_length = seq_length
        self.embeddings_count = len(vocab_dict) + 2
        self.num_classes = num_classes

        # Model layers
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count, output_dim=128, input_length=seq_length)
        self.conv1d = tf.keras.layers.Conv1D(
            filters=64, kernel_size=5, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=True))
        self.dense_query = tf.keras.layers.Dense(64)
        self.attention = tf.keras.layers.Attention()
        self.concatenate = tf.keras.layers.Concatenate()
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(
            num_classes, activation='softmax')

    def call(self, inputs):
        # Forward pass
        x = self.embedding(inputs)
        x = self.conv1d(x)
        x = self.dropout1(x)
        x = self.bilstm(x)
        # query = self.dense_query(x)
        # attention = self.attention([query, x])
        # x = self.concatenate([x, attention])
        x = self.global_max_pool(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        outputs = self.output_layer(x)
        return outputs
