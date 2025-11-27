import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import constraints, initializers, regularizers


@tf.keras.utils.register_keras_serializable(package="dlomix")
class DecoderAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, time_steps, **kwargs):
        super().__init__(**kwargs)
        self.time_steps = time_steps
        self.permute = None
        self.dense = None
        self.multiply = None

    def build(self, input_shape):
        self.permute = tf.keras.layers.Permute((2, 1))
        self.dense = tf.keras.layers.Dense(self.time_steps, activation="softmax")
        self.multiply = tf.keras.layers.Multiply()
        super().build(input_shape)

    def call(self, inputs):
        x = self.permute(inputs)
        x = self.dense(x)
        x = self.permute(x)
        x = self.multiply([inputs, x])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"time_steps": self.time_steps})
        return config

    # No from_config needed! Default works fine.


@tf.keras.utils.register_keras_serializable(package="dlomix")
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
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self.W = None
        self.b = None
        self.u = None

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer=self.init,
            name=f"{self.name}_W",
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )
        if self.bias:
            self.b = self.add_weight(
                shape=(input_shape[1],),
                initializer="zero",
                name=f"{self.name}_b",
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )
        if self.context:
            self.u = self.add_weight(
                shape=(input_shape[-1],),
                initializer=self.init,
                name=f"{self.name}_u",
                regularizer=self.u_regularizer,
                constraint=self.u_constraint,
            )
        super().build(input_shape)

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
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bias": self.bias,
                "context": self.context,
                "W_regularizer": regularizers.serialize(self.W_regularizer),
                "b_regularizer": regularizers.serialize(self.b_regularizer),
                "u_regularizer": regularizers.serialize(self.u_regularizer),
                "W_constraint": constraints.serialize(self.W_constraint),
                "b_constraint": constraints.serialize(self.b_constraint),
                "u_constraint": constraints.serialize(self.u_constraint),
            }
        )
        return config
