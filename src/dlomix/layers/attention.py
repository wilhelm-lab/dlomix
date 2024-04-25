import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import constraints, initializers, regularizers


class DecoderAttentionLayer(tf.keras.layers.Layer):
    """
    Decoder attention layer.

    Parameters
    ----------
    time_steps : int
        Number of time steps in the input data.
    """

    def __init__(self, time_steps):
        super(DecoderAttentionLayer, self).__init__()
        self.time_steps = time_steps

    def build(self, input_shape):
        """
        Build the layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        """
        self.permute = tf.keras.layers.Permute((2, 1))
        self.dense = tf.keras.layers.Dense(self.time_steps, activation="softmax")
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        """
        Perform the forward pass of the layer.

        Parameters
        ----------
        inputs : tensor
            Input tensor.

        Returns
        -------
        tensor
            Output tensor.
        """
        x = self.permute(inputs)
        x = self.dense(x)
        x = self.permute(x)
        x = self.multiply([inputs, x])
        return x


class AttentionLayer(tf.keras.layers.Layer):
    """
    Attention layer.

    Parameters
    ----------
    context : bool, optional
        Whether to use context or not. Defaults to False.
    W_regularizer : str, optional
        Regularizer for the weights. Defaults to None.
    b_regularizer : str, optional
        Regularizer for the bias. Defaults to None.
    u_regularizer : str, optional
        Regularizer for the context. Defaults to None.
    W_constraint : str, optional
        Constraint for the weights. Defaults to None.
    b_constraint : str, optional
        Constraint for the bias. Defaults to None.
    u_constraint : str, optional
        Constraint for the context. Defaults to None.
    bias : bool, optional
        Whether to use bias or not. Defaults to True.
    """

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
        """
        Build the layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        """
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
        """
        Compute the mask for the layer.

        Parameters
        ----------
        input : tensor
            Input tensor.
        input_mask : tensor, optional
            Input mask tensor. Defaults to None.

        Returns
        -------
        tensor
            Mask tensor.
        """
        return None

    def call(self, x, mask=None):
        """
        Perform the forward pass of the layer.

        Parameters
        ----------
        x : tensor
            Input tensor.
        mask : tensor, optional
            Mask tensor. Defaults to None.

        Returns
        -------
        tensor
            Output tensor.
        """
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
        """
        Compute the output shape of the layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.

        Returns
        -------
        tuple
            Shape of the output tensor.
        """
        return input_shape[0], input_shape[-1]

    def get_config(self):
        """
        Get the configuration of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """
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
