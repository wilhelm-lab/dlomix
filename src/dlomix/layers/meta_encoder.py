import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="dlomix")
class MetaEncoder(tf.keras.layers.Layer):
    """
    Encoder for experimental metadata inputs.

    This layer concatenates multiple metadata inputs (such as collision energy,
    precursor charge, fragmentation type) and processes them through a dense
    layer with dropout for regularization.

    Parameters
    ----------
    output_dim : int, optional
        Size of the dense layer output. Defaults to 32.
    dropout_rate : float, optional
        Dropout rate for regularization. Defaults to 0.2.

    Notes
    -----
    The layer expects a list of input tensors which are concatenated along
    the last axis before processing. This is commonly used to encode
    experimental conditions in mass spectrometry prediction models.

    Example
    -------
    >>> meta_encoder = MetaEncoder(output_dim=32, dropout_rate=0.3)
    >>> # Inputs: [collision_energy, precursor_charge, fragmentation_type]
    >>> encoded = meta_encoder([ce_tensor, charge_tensor, frag_tensor])
    """

    def __init__(self, output_dim=32, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Initialize as None - will be created in build()
        self.concat = None
        self.dense = None
        self.dropout = None

    def build(self, input_shape):
        self.concat = tf.keras.layers.Concatenate(name="meta_concat")
        self.dense = tf.keras.layers.Dense(
            self.output_dim, name="meta_dense"
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, name="meta_dropout")
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Encode metadata inputs.

        Parameters
        ----------
        inputs : list of tf.Tensor
            List of input tensors to concatenate and encode. Each tensor should
            have shape (batch_size, feature_dim).
        training : bool, optional
            Whether the layer is in training mode (affects dropout).

        Returns
        -------
        tf.Tensor
            Encoded metadata tensor of shape (batch_size, output_dim).
        """
        x = self.concat(inputs)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
