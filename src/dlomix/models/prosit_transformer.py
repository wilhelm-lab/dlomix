import logging

import tensorflow as tf

from ..layers import (
    MetaEncoder,
    PositionalEmbedding,
    TransformerEncoder,
)

logger = logging.getLogger("dlomix.models.prosit_transformer")

_ALPHABET_ORDERED = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET = {k: v for v, k in enumerate(_ALPHABET_ORDERED, start=1)}


@tf.keras.utils.register_keras_serializable(package="dlomix")
class PrositTransformerIntensityPredictor(tf.keras.Model):
    """
    Prosit Transformer model for intensity prediction with gain/loss modifications.

    This model uses transformer encoders instead of recurrent layers to capture sequence
    dependencies, combined with modification gain/loss features and experimental metadata.

    Parameters
    ----------
    embedding_output_dim : int, optional
        Size of the transformer embeddings. Defaults to 64.
    alphabet : dict, optional
        Dictionary mapping for the alphabet (amino acids). Defaults to ALPHABET.
    dropout_rate : float, optional
        Probability to use for dropout layers in the meta encoder. Defaults to 0.2.
    num_heads : int, optional
        Number of attention heads in the transformer. Defaults to 16.
    ff_dim : int, optional
        Dimension of the feed-forward network in the transformer. Defaults to 32.
    transformer_dropout : float, optional
        Dropout rate within transformer layers. Defaults to 0.1.
    num_transformers : int, optional
        Number of transformer encoder blocks to stack. Defaults to 6.
    input_keys : dict, optional
        Dictionary mapping for the input keys to look for in the input dict. Defaults to None,
        which uses DEFAULT_INPUT_KEYS.
    meta_data_keys : dict, optional
        Dictionary of keys for the meta data inputs to use. Defaults to None, which uses
        META_DATA_KEYS.

    Attributes
    ----------
    DEFAULT_INPUT_KEYS : dict
        Default mapping of input keys for various inputs including sequence, collision energy,
        precursor charge, fragmentation type, and instrument type.
    META_DATA_KEYS : list
        List of metadata keys used for experimental conditions.

    Notes
    -----
    The model architecture consists of:
    - Positional embeddings for sequence representation
    - Gain/loss modification encoder
    - Meta data encoder for experimental conditions
    - Multi-head transformer encoder blocks
    - Dense layers for final prediction
    """

    DEFAULT_INPUT_KEYS = {
        "SEQUENCE_KEY": "modified_sequence",
        "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
        "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
        "FRAGMENTATION_TYPE_KEY": "fragmentation_onehot",
        "INSTRUMENT_TYPE_KEY": "instrument_onehot",
    }

    META_DATA_KEYS = [
        "COLLISION_ENERGY_KEY",
        "PRECURSOR_CHARGE_KEY",
        "FRAGMENTATION_TYPE_KEY",
        "INSTRUMENT_TYPE_KEY",
    ]

    def __init__(
        self,
        embedding_output_dim=64,
        alphabet=None,
        dropout_rate=0.2,
        num_heads=16,
        ff_dim=32,
        transformer_dropout=0.1,
        num_transformers=6,
        input_keys=None,
        meta_data_keys=None,
        **kwargs,
    ):
        super(PrositTransformerIntensityPredictor, self).__init__(**kwargs)

        # Store hyperparameters
        self.alphabet = alphabet if alphabet is not None else ALPHABET
        self.embedding_output_dim = embedding_output_dim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.transformer_dropout = transformer_dropout
        self.num_transformers = num_transformers

        # Set input keys
        if input_keys is None:
            self.input_keys = self.DEFAULT_INPUT_KEYS.copy()
        else:
            self.input_keys = {**self.DEFAULT_INPUT_KEYS, **input_keys}

        # Set meta data keys
        if meta_data_keys is None:
            self.meta_data_keys = {
                k: self.input_keys[k] for k in self.META_DATA_KEYS if k in self.input_keys
            }
        else:
            self.meta_data_keys = meta_data_keys

        # Calculate embeddings count based on alphabet size
        self.embeddings_count = len(self.alphabet) + 2

        # Build model components
        self._build_embedding_layers()
        self._build_meta_encoder()
        self._build_transformer_encoder()
        self._build_gain_loss_encoder()
        self._build_output_layers()

    def _build_embedding_layers(self):
        """Build the positional embedding layer."""
        self.pos_embedding = PositionalEmbedding(self.embeddings_count, 32)

    def _build_meta_encoder(self):
        """Build the metadata encoder for experimental conditions."""
        self.meta_encoder = MetaEncoder(output_dim=32, dropout_rate=self.dropout_rate)

    def _build_transformer_encoder(self):
        """Build the transformer encoder stack."""
        self.transformer_encoder = TransformerEncoder(
            embed_dim=self.embedding_output_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            rate=self.transformer_dropout,
            num_transformers=self.num_transformers,
        )

    def _build_gain_loss_encoder(self):
        """Build the gain/loss modification encoder."""
        self.gain_loss_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Concatenate(name="gain_loss"),
                tf.keras.layers.Dense(256, name="gain_loss_dense_1"),
                tf.keras.layers.Dense(32, name="gain_loss_dense_2"),
            ],
            name="gain_loss_encoder",
        )

    def _build_output_layers(self):
        """Build the output prediction layers."""
        self.output_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, name="dense_1"),
                tf.keras.layers.Dense(256, name="dense_2"),
                tf.keras.layers.BatchNormalization(name="batch_norm"),
                tf.keras.layers.LeakyReLU(name="leaky_relu"),
                tf.keras.layers.Dense(174, name="output_dense"),
            ],
            name="output_layers",
        )

    def _build_output_layers(self):
        """Split output layers into pre- and post-transformer blocks."""
        # Layers applied before transformer
        self.pre_transformer_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, name="dense_1"),
            ],
            name="pre_transformer_layers",
        )

        # Layers applied after transformer
        self.post_transformer_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, name="dense_2"),
                tf.keras.layers.BatchNormalization(name="batch_norm"),
                tf.keras.layers.LeakyReLU(name="leaky_relu"),
                tf.keras.layers.Dense(174, name="output_dense"),
            ],
            name="post_transformer_layers",
        )


    def call(self, inputs, **kwargs):
        """
        Forward pass of the model.

        Parameters
        ----------
        inputs : dict
            Dictionary containing all required inputs including sequence, metadata,
            and gain/loss modifications.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            Predicted intensities with shape (batch_size, 174).
        """
        # Extract inputs
        peptides_in = inputs.get(self.input_keys["SEQUENCE_KEY"])
        collision_energy_in = inputs.get(self.meta_data_keys["COLLISION_ENERGY_KEY"])
        precursor_charge_in = inputs.get(self.meta_data_keys["PRECURSOR_CHARGE_KEY"])
        fragm_method_in = inputs.get(self.meta_data_keys["FRAGMENTATION_TYPE_KEY"])
        loss_in = inputs["mod_loss"]
        gain_in = inputs["mod_gain"]

        # Validate required inputs
        if peptides_in is None:
            raise ValueError("Missing required input: sequence")
        if collision_energy_in is None:
            raise ValueError("Missing required input: collision_energy")
        if precursor_charge_in is None:
            raise ValueError("Missing required input: precursor_charge")
        if fragm_method_in is None:
            raise ValueError("Missing required input: fragmentation_type")

        # Reshape inputs if needed
        if len(collision_energy_in.shape) == 1:
            collision_energy_in = tf.expand_dims(collision_energy_in, axis=-1)

        if len(fragm_method_in.shape) == 1:
            fragm_method_in = tf.expand_dims(fragm_method_in, axis=1)

        # Encode gain/loss modifications
        gain_loss = self.gain_loss_encoder([loss_in, gain_in])

        # Embed peptide sequence with positional encoding
        x = self.pos_embedding(peptides_in)

        # Concatenate sequence embeddings with gain/loss features
        x = tf.keras.layers.Concatenate(name="peptides_gain_loss")([x, gain_loss])

        # Encode metadata
        precursor_charge_in = tf.cast(precursor_charge_in, tf.float32)
        encoded_meta = self.meta_encoder(
            [collision_energy_in, precursor_charge_in, fragm_method_in]
        )

        # Expand and tile metadata to match sequence length
        encoded_meta = tf.expand_dims(encoded_meta, axis=-1)
        encoded_meta = tf.tile(encoded_meta, [1, 1, 64])

        # Concatenate sequence with metadata
        x = tf.concat([x, encoded_meta], axis=-1)

        # Pass through dense layer before transformer
        x = self.pre_transformer_layers(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Apply remaining output layers
        x = self.post_transformer_layers(x)

        # Pool across sequence dimension
        x = tf.reduce_mean(x, axis=1)

        return x

    def get_config(self):
        """
        Get the configuration of the model for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing all parameters needed to reconstruct the model.
        """
        config = super().get_config()
        config.update(
            {
                "embedding_output_dim": self.embedding_output_dim,
                "alphabet": self.alphabet,
                "dropout_rate": self.dropout_rate,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "transformer_dropout": self.transformer_dropout,
                "num_transformers": self.num_transformers,
                "input_keys": self.input_keys,
                "meta_data_keys": self.meta_data_keys,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Recreate model from configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        PrositTransformerIntensityPredictor
            Reconstructed model instance.
        """
        return cls(**config)