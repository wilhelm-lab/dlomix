import tensorflow as tf
import tf.keras.backend as K
from tf.keras import constraints, initializers, regularizers

from ..constants import ALPHABET_UNMOD

"""
--------------------------------------------------------------
TASK MODELS
DominantChargeStatePredictor

ObservedChargeStatePredictor

ChargeStateProportionPredictor

--------------------------------------------------------------
"""


class DominantChargeStatePredictor(tf.keras.Model):
    """
    Task 1
    Predict the dominant charge state
    Prosite Charge-State Adaption
    """

    def __init__(
        self,
        embedding_output_dim=16,
        seq_length=30,
        vocab_dict=ALPHABET_UNMOD,
        dropout_rate=0.5,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
        num_classes=6,
    ):
        super(DominantChargeStatePredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(vocab_dict) + 2

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_output_dim,
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

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        return x


class ObservedChargeStatePredictor(tf.keras.Model):
    """
    Task 2
    Predict all observed charge states
    """

    def __init__(
        self,
        embedding_output_dim=16,
        seq_length=30,
        vocab_dict=ALPHABET_UNMOD,
        dropout_rate=0.5,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
        num_classes=6,
    ):
        super(ObservedChargeStatePredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(vocab_dict) + 2

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_output_dim,
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

        self.output_layer = tf.keras.layers.Dense(num_classes, activation="sigmoid")

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

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        return x


class ChargeStateDistributionPredictor(tf.keras.Model):
    """
    Task 3
    Predict the proportion of all observed charge states
    """

    def __init__(
        self,
        embedding_output_dim=16,
        seq_length=30,
        vocab_dict=ALPHABET_UNMOD,
        dropout_rate=0.5,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
        num_classes=6,  # TODO number of charge states etc
    ):
        super(ChargeStateDistributionPredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(vocab_dict) + 2

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_output_dim,
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

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        x = self.attention(x)
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
