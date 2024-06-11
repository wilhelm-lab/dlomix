import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations, constraints, initializers, regularizers

from ..constants import ALPHABET_UNMOD

"""
----------------------------------------------
TEST/EXPERIMENTAL MODELS
----------------------------------------------
"""


class LSTMTest(tf.keras.Model):
    """
    Predicts the dominant charge state incorporating bidirectional LSTM layers.
    """

    def __init__(
        self,
        embedding_dim=16,
        seq_length=30,
        vocab_dict=ALPHABET_UNMOD,
        num_classes=7,
        lstm_units=64,
    ):
        super(LSTMTest, self).__init__()
        # Model parameters
        self.embedding_dim = embedding_dim
        self.embeddings_count = len(vocab_dict) + 2
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.lstm_units = lstm_units

        # Model layers
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=self.embedding_dim,
            input_length=self.seq_length,
        )
        self.bidirectional_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.lstm_units, return_sequences=True)
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(self.seq_length, activation="relu")
        self.dense2 = tf.keras.layers.Dense(self.num_classes, activation="softmax")

    def call(self, inputs):
        # Forward pass
        x = self.embedding(inputs)
        x = self.bidirectional_lstm(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class LayerTestChargePredictor(tf.keras.Model):
    def __init__(
        self, seq_length=30, vocab_dict=ALPHABET_UNMOD, num_classes=7, lstm_units=64
    ):
        super(LayerTestChargePredictor, self).__init__()
        # Model parameters
        self.seq_length = seq_length
        self.embeddings_count = len(vocab_dict) + 2
        self.num_classes = num_classes

        # Model layers
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count, output_dim=128, input_length=seq_length
        )
        self.conv1d = tf.keras.layers.Conv1D(
            filters=64, kernel_size=5, activation="relu"
        )
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=True)
        )
        self.dense_query = tf.keras.layers.Dense(64)
        self.attention = tf.keras.layers.Attention()
        self.concatenate = tf.keras.layers.Concatenate()
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        # Forward pass
        print("Shape of inputs: ", inputs.shape)
        x = self.embedding(inputs)
        print("Shape after embedding: ", x.shape)
        # x = self.conv1d(x)
        # print('Shape after conv1d: ', x.shape)
        # x = self.dropout1(x)
        # print('Shape after dropout: ', x.shape)
        x = self.bilstm(x)
        print("Shape after bilstm: ", x.shape)
        attention = self.attention([x, x])
        print("Shape after attention: ", attention.shape)
        x = self.flatten(attention)
        print("Shape after flatten: ", x.shape)
        x = self.dense1(x)
        print("Shape after dense1: ", x.shape)
        # query = self.dense_query(x)
        # print('Shape after query: ', query.shape)
        # attention = self.attention([query, x])
        # print('Shape after attention: ', attention.shape)
        # x = self.concatenate([x, attention])
        # print('Shape after concatenate: ', x.shape)
        # x = self.global_max_pool(x)
        # print('Shape after golbal max pool: ', x.shape)
        # x = self.dense1(x)
        # print('Shape after dens1: ', x.shape)
        # x = self.dropout2(x)
        # print('Shape after dropout 2: ', x.shape)
        outputs = self.output_layer(x)
        print("Shape of output: ", outputs.shape)
        return outputs


class PrositChargeStateAdaption(tf.keras.Model):
    def __init__(
        self,
        embedding_dim=64,
        seq_length=30,
        vocab_dict=ALPHABET_UNMOD,
        num_classes=6,
        dropout_rate=0.5,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
    ):
        super(PrositChargeStateAdaption, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(vocab_dict) + 2
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_dim,
            input_length=seq_length,
        )

        self._build_encoder()

        self.attention = AttentionLayer()

        self.regressor = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.regressor_layer_size, activation="relu"),
                tf.keras.layers.Dropout(rate=self.latent_dropout_rate),
            ]
        )

        self.dense1 = tf.keras.layers.Dense(64, activation="relu")

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU())

        self.output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")

    def _build_encoder(self):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(
                        units=self.recurrent_layers_sizes[0], return_sequences=True
                    )
                ),
                tf.keras.layers.Dropout(rate=self.dropout_rate),
                tf.keras.layers.GRU(
                    units=self.recurrent_layers_sizes[1], return_sequences=True
                ),
                tf.keras.layers.Dropout(rate=self.dropout_rate),
            ]
        )

    def call(self, inputs, **kwargs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        x = self.attention(x)
        ############################################################
        x = self.flatten(x)
        x = self.dense1(x)
        ############################################################
        x = self.regressor(x)
        x = self.output_layer(x)
        return x


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        context=False,
        W_regularizer=None,
        b_regularizer=None,
        u_regularizer=None,
        W_constraint=None,
        b_constraint=None,
        u_constraint=None,
        bias=True,
        **kwargs
    ):
        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.bias = bias
        self.context = context
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer=self.init,
            name="{}_W".format(self.name),
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )
        if self.bias:
            self.b = self.add_weight(
                shape=(input_shape[1],),
                initializer="zero",
                name="{}_b".format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )
        else:
            self.b = None
        if self.context:
            self.u = self.add_weight(
                shape=(input_shape[-1],),
                initializer=self.init,
                name="{}_u".format(self.name),
                regularizer=self.u_regularizer,
                constraint=self.u_constraint,
            )

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        a = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)
        if self.bias:
            a += self.b
        a = K.tanh(a)
        if self.context:
            a = K.squeeze(K.dot(x, K.expand_dims(self.u)), axis=-1)
        a = K.exp(a)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = {
            "bias": self.bias,
            "context": self.context,
            "W_regularizer": regularizers.serialize(self.W_regularizer),
            "b_regularizer": regularizers.serialize(self.b_regularizer),
            "u_regularizer": regularizers.serialize(self.u_regularizer),
            "W_constraint": constraints.serialize(self.W_constraint),
            "b_constraint": constraints.serialize(self.b_constraint),
            "u_constraint": constraints.serialize(self.u_constraint),
        }
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DominantChargeStatePredictorTest(tf.keras.Model):
    def __init__(
        self,
        embedding_dim=64,
        seq_length=30,
        vocab_dict=ALPHABET_UNMOD,
        num_classes=7,
        dropout_rate=0.5,
    ):
        super(DominantChargeStatePredictorTest, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings_count = len(vocab_dict) + 2
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=self.embedding_dim,
            input_length=self.seq_length,
        )
        self.conv1d = tf.keras.layers.Conv1D(
            filters=128, kernel_size=3, padding="same", activation="relu"
        )
        self.rnn_layer1 = tf.keras.layers.GRU(256, return_sequences=True)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.rnn_layer2 = tf.keras.layers.GRU(128, return_sequences=True)
        self.attention = AttentionLayer()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.conv1d(x)
        x = self.rnn_layer1(x)
        x = self.dropout1(x)
        x = self.rnn_layer2(x)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x
