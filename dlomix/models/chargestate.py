import tensorflow as tf
from ..constants import ALPHABET_UNMOD


'''
--------------------------------------------------------------
TASK MODELS
TODO: DominantChargeStatePredictor
- add sequence specific layers like LSTM, Conv1D, etc

TODO: ObservedChargeStatePredictor
- add sequence specific layers like LSTM, Conv1D, etc
- take additional features into account 

TODO: ChargeStateProportionPredictor
- implement
--------------------------------------------------------------
'''


class DominantChargeStatePredictor(tf.keras.Model):
    '''
    Task 1 
    Predict the dominant charge state
    '''

    def __init__(self, embedding_dim=16, seq_length=30, vocab_dict=ALPHABET_UNMOD, num_classes=7):
        # inititalize the model layers
        super(DominantChargeStatePredictor, self).__init__()
        # Model parameters
        self.embedding_dim = embedding_dim
        self.embeddings_count = len(vocab_dict) + 2
        self.seq_length = seq_length
        self.num_classes = num_classes

        # Model layers
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count, output_dim=self.embedding_dim, input_length=self.seq_length)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(self.seq_length, activation="relu")
        self.dense2 = tf.keras.layers.Dense(
            self.num_classes, activation="softmax")

    def call(self, inputs):
        # Forward pass
        x = self.embedding(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        # debug by printing shape of inputs/outputs ect
        return x


class ObservedChargeStatePredictor(tf.keras.Model):
    '''
    Task 2
    Predict all observed charge states
    '''

    def __init__(self, embedding_dim=16, seq_length=30, vocab_dict=ALPHABET_UNMOD, num_classes=7):
        super(ObservedChargeStatePredictor, self).__init__()
        # Model parameters
        self.embedding_dim = embedding_dim
        self.embeddings_count = len(vocab_dict) + 2
        self.seq_length = seq_length
        self.num_classes = num_classes

        # Model layers
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count, output_dim=self.embedding_dim, input_length=self.seq_length)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(self.seq_length, activation="relu")
        self.dense2 = tf.keras.layers.Dense(num_classes, activation="sigmoid")

    def call(self, inputs):
        # Forward pass
        x = self.embedding(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class ChargeStateProportionPredictor(tf.keras.Model):
    '''
    Task 3
    Predict the proportion of all observed charge states
    '''
    pass
