import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="dlomix")
class TransformerBlock(tf.keras.layers.Layer):
    """
    Single transformer encoder block with multi-head attention and feed-forward network.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings and output.
    num_heads : int
        Number of attention heads.
    ff_dim : int
        Dimensionality of the feed-forward network hidden layer.
    rate : float, optional
        Dropout rate. Defaults to 0.1.

    Notes
    -----
    The block consists of:
    - Multi-head self-attention with residual connection and layer normalization
    - Feed-forward network with residual connection and layer normalization
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        # Initialize layer attributes as None - will be created in build()
        self.att = None
        self.ffn = None
        self.layernorm1 = None
        self.layernorm2 = None
        self.dropout1 = None
        self.dropout2 = None

    def build(self, input_shape):
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim, name="multi_head_attention"
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.ff_dim, activation="relu", name="ffn_dense_1"),
                tf.keras.layers.Dense(self.embed_dim, name="ffn_dense_2"),
            ],
            name="feed_forward",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="layer_norm_1"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="layer_norm_2"
        )
        self.dropout1 = tf.keras.layers.Dropout(self.rate, name="dropout_1")
        self.dropout2 = tf.keras.layers.Dropout(self.rate, name="dropout_2")
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Forward pass through the transformer block.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, seq_len, embed_dim).
        training : bool, optional
            Whether the layer is in training mode.

        Returns
        -------
        tf.Tensor
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="dlomix")
class TransformerEncoder(tf.keras.layers.Layer):
    """
    Stack of transformer encoder blocks.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads in each transformer block.
    ff_dim : int
        Dimensionality of the feed-forward network hidden layer in each block.
    rate : float, optional
        Dropout rate for each transformer block. Defaults to 0.1.
    num_transformers : int, optional
        Number of transformer blocks to stack. Defaults to 2.

    Notes
    -----
    This layer stacks multiple TransformerBlock layers sequentially to build
    a deeper transformer encoder architecture.
    """

    def __init__(
        self, embed_dim, num_heads, ff_dim, rate=0.1, num_transformers=2, **kwargs
    ):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.num_transformers = num_transformers

        # Initialize as None - will be created in build()
        self.transformer_blocks = None

    def build(self, input_shape):
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                rate=self.rate,
                name=f"transformer_block_{i}",
            )
            for i in range(self.num_transformers)
        ]
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Forward pass through all transformer blocks.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, seq_len, embed_dim).
        training : bool, optional
            Whether the layer is in training mode.

        Returns
        -------
        tf.Tensor
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        x = inputs
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
                "num_transformers": self.num_transformers,
            }
        )
        return config
