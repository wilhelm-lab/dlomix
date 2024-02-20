import tensorflow as tf
from ..constants import ALPHABET_UNMOD


class DominantChargeStatePredictor(tf.keras.Model):
    def __init__(self, embedding_dim=16, seq_length=30, vocab_dict=ALPHABET_UNMOD, num_classes=6):
        # inititalize the model layers
        super(DominantChargeStatePredictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings_count = len(vocab_dict) + 2
        self.seq_length = seq_length
        self.num_classes = num_classes

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count, output_dim=self.embedding_dim, input_length=self.seq_length)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            self.seq_length, activation="relu")
        self.dense2 = tf.keras.layers.Dense(
            self.num_classes, activation="softmax")

    def call(self, inputs):
        # define forward pass
        # debug by printing shape of inputs ect
        x = self.embedding(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
