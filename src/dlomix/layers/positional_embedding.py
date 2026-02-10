import tensorflow as tf
import numpy as np


def positional_encoding(length, depth, pos_scaling=8):
    """
    Generate sinusoidal positional encodings.

    Parameters
    ----------
    length : int
        Maximum sequence length.
    depth : int
        Dimensionality of the encoding (will be divided by 2 for sin/cos).
    pos_scaling : float, optional
        Scaling factor for position encoding frequencies. Defaults to 8.

    Returns
    -------
    tf.Tensor
        Positional encoding tensor of shape (length, depth).

    Notes
    -----
    Uses sinusoidal functions with different frequencies to encode position information.
    The encoding uses sine for even indices and cosine for odd indices.
    """
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    angle_rates = 1 / ((pos_scaling * length) ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


@tf.keras.utils.register_keras_serializable(package="dlomix")
class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Combines token embeddings with sinusoidal positional encodings.

    This layer first embeds discrete tokens and then adds positional information
    using sinusoidal encodings, scaled by the square root of the embedding dimension.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (number of unique tokens).
    d_model : int
        Dimensionality of the embedding space.
    max_length : int, optional
        Maximum sequence length for positional encoding. Defaults to 512.
    pos_scaling : float, optional
        Scaling factor for position encoding frequencies. Defaults to 8.

    Notes
    -----
    The layer performs the following operations:
    1. Embeds input tokens to d_model dimensions
    2. Scales embeddings by sqrt(d_model)
    3. Adds positional encodings to the scaled embeddings

    The embedding layer uses mask_zero=True to handle variable-length sequences.
    """

    def __init__(
        self, vocab_size, d_model, max_length=512, pos_scaling=8, **kwargs
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.pos_scaling = pos_scaling

        # Initialize as None - will be created in build()
        self.embedding = None
        self.pos_encoding = positional_encoding(
            length=max_length, depth=d_model, pos_scaling=pos_scaling
        )

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, self.d_model, mask_zero=True, name="token_embedding"
        )
        super().build(input_shape)

    def compute_mask(self, *args, **kwargs):
        """
        Compute mask for variable-length sequences.

        Returns
        -------
        tf.Tensor or None
            Boolean mask tensor indicating which positions are valid (not padding).
        """
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, inputs):
        """
        Apply embedding and positional encoding to inputs.

        Parameters
        ----------
        inputs : tf.Tensor
            Integer tensor of token indices with shape (batch_size, seq_len).

        Returns
        -------
        tf.Tensor
            Embedded and position-encoded tensor of shape (batch_size, seq_len, d_model).
        """
        length = tf.shape(inputs)[1]
        x = self.embedding(inputs)
        # Scale embeddings by sqrt(d_model) as in the original Transformer paper
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # Add positional encoding
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "max_length": self.max_length,
                "pos_scaling": self.pos_scaling,
            }
        )
        return config
