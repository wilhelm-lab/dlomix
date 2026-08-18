import warnings

import tensorflow as tf

from ..constants import ALPHABET_UNMOD
from ..layers.attention import AttentionLayer
from ._alphabet import validate_alphabet_size

"""
This module contains a deep learning model for precursor charge state prediction, inspired by Prosit's architecture.
The model is provided in three flavours of predicting precursor charge states:

1. Dominant Charge State Prediction:
   - Task: Predict the dominant charge state of a given peptide sequence.
   - Model: Uses a multi-class classification approach to predict the most likely charge state.

2. Observed Charge State Prediction:
   - Task: Predict the observed charge states for a given peptide sequence.
   - Model: Uses a multi-label classification approach to predict all possible charge states.

3. Relative Charge State Prediction:
   - Task: Predict the proportion of each charge state for a given peptide sequence.
   - Model: Uses a regression approach to predict the proportion of each charge state.
"""


@tf.keras.utils.register_keras_serializable(package="dlomix")
class ChargeStatePredictor(tf.keras.Model):
    """
    Precursor Charge State Prediction Model for predicting either:
    * the dominant charge state or
    * all observed charge states or
    * the relative charge state distribution
    of a peptide sequence.

    Args:
        embedding_output_dim (int): The size of the embedding output dimension. Defaults to 16.
        seq_length (int): The length of the input sequence. Defaults to 30.
        alphabet (dict): Dictionary mapping for the alphabet (the amino acids in this case). Defaults to ALPHABET_UNMOD.
        dropout_rate (float): The dropout rate used in the encoder layers. Defaults to 0.5.
        latent_dropout_rate (float): The dropout rate for the latent space. Defaults to 0.1.
        recurrent_layers_sizes (tuple): The sizes of the recurrent layers. Defaults to (256, 512).
        regressor_layer_size (int): The size of the regressor layer. Defaults to 512.
        num_classes (int): The number of classes for the output corresponding to charge states available in the data. Defaults to 6.
        model_flavour (str): The type of precursor charge state prediction to be done.
            Can be either "dominant", "observed" or "relative".
            Defaults to "relative".
    """

    def __init__(
        self,
        embedding_output_dim=16,
        seq_length=30,
        alphabet=ALPHABET_UNMOD,
        dropout_rate=0.5,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
        num_classes=6,
        model_flavour="relative",
        **kwargs,
    ):
        super(ChargeStatePredictor, self).__init__(**kwargs)

        # the vocabulary already carries the padding and unknown tokens, so its
        # length is exactly the number of embedding rows needed
        validate_alphabet_size(alphabet, type(self).__name__)
        self.embeddings_count = len(alphabet)

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = tuple(recurrent_layers_sizes)
        self.embedding_output_dim = embedding_output_dim
        self.seq_length = seq_length
        self.alphabet = dict(alphabet)
        self.num_classes = num_classes
        self.model_flavour = model_flavour

        if model_flavour == "relative":
            # regression problem
            self.final_activation = "linear"
        elif model_flavour == "observed":
            # multi-label multi-class classification problem
            self.final_activation = "sigmoid"
        elif model_flavour == "dominant":
            # multi-class classification problem
            self.final_activation = "softmax"
        else:
            warnings.warn(f"{model_flavour} not available")
            exit

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_output_dim,
        )
        self._build_encoder()

        self.attention = AttentionLayer()

        self.regressor = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.regressor_layer_size, activation="relu"),
                tf.keras.layers.Dropout(rate=self.latent_dropout_rate),
            ]
        )

        self.output_layer = tf.keras.layers.Dense(
            num_classes, activation=self.final_activation
        )

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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_output_dim": self.embedding_output_dim,
                "seq_length": self.seq_length,
                "alphabet": self.alphabet,
                "dropout_rate": self.dropout_rate,
                "latent_dropout_rate": self.latent_dropout_rate,
                "recurrent_layers_sizes": list(self.recurrent_layers_sizes),
                "regressor_layer_size": self.regressor_layer_size,
                "num_classes": self.num_classes,
                "model_flavour": self.model_flavour,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "recurrent_layers_sizes" in config and isinstance(
            config["recurrent_layers_sizes"], list
        ):
            config["recurrent_layers_sizes"] = tuple(config["recurrent_layers_sizes"])
        return cls(**config)
