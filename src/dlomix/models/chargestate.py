import warnings

import tensorflow as tf

from ..constants import ALPHABET_UNMOD
from ..layers.attention import AttentionLayer

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
    ):
        super(ChargeStatePredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(alphabet) + 1

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes

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
