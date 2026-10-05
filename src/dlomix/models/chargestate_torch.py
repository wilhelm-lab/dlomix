from collections import OrderedDict

import torch.nn as nn

from ..constants import ALPHABET_UNMOD
from ..layers.attention_torch import AttentionLayer
from ..layers.bi_gru_seq_encoder_torch import BiGRUSequentialEncoder
from ..layers.keras_initializers_torch import init_like_keras
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


class ChargeStatePredictor(nn.Module):
    """
    Precursor Charge State Prediction Model for predicting either:
    * the dominant charge state or
    * all observed charge states or
    * the relative charge state distribution
    of a peptide sequence.

    batch_first used internally (batch, sequence, feature)

    Args:
        embedding_output_dim (int): The size of the embedding output dimension. Defaults to 16.
        seq_length (int): The length of the input sequence. Defaults to 30. Not used to
            build the model (as in TensorFlow): the layers take the padded length from
            the first input.
        alphabet (dict): Dictionary mapping for the alphabet (the amino acids in this case). Defaults to ALPHABET_UNMOD.
        dropout_rate (float): The dropout rate used in the encoder layers. Defaults to 0.5.
        latent_dropout_rate (float): The dropout rate for the latent space. Defaults to 0.1.
        recurrent_layers_sizes (tuple): The sizes of the recurrent layers. Defaults to (256, 512).
        regressor_layer_size (int): The size of the regressor layer. Defaults to 512.
        num_classes (int): The number of classes for the output corresponding to charge states available in the data. Defaults to 6.
        model_flavour (str): The type of precursor charge state prediction to be done.
            Can be either "dominant", "observed" or "relative". Defaults to "relative".
            As in TensorFlow, "dominant" returns softmax probabilities, so train it with
            ``nn.NLLLoss()`` on ``torch.log(probabilities)``, not with ``nn.CrossEntropyLoss``
            (which expects logits).
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

        # the vocabulary already carries the padding and unknown tokens, so its
        # length is exactly the number of embedding rows needed (as in Keras)
        validate_alphabet_size(alphabet, type(self).__name__)
        self.embeddings_count = len(alphabet)
        self.seq_length = seq_length

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes

        # same output on both backends: probabilities for the classification flavours
        final_activations = {
            "relative": nn.Identity,  # regression
            "observed": nn.Sigmoid,  # multi-label classification
            "dominant": lambda: nn.Softmax(dim=-1),  # multi-class classification
        }
        if model_flavour not in final_activations:
            raise ValueError(
                f"Unknown model_flavour {model_flavour!r}, expected one of "
                f"{sorted(final_activations)}."
            )
        self.model_flavour = model_flavour
        self.final_activation = final_activations[model_flavour]()

        self.embedding = nn.Embedding(
            num_embeddings=self.embeddings_count,
            embedding_dim=embedding_output_dim,
        )

        self.encoder = BiGRUSequentialEncoder(
            embedding_output_dim, self.recurrent_layers_sizes, self.dropout_rate
        )

        # the per-position attention bias is sized from the first input, as in Keras
        self.attention = AttentionLayer(feature_dim=self.recurrent_layers_sizes[1])

        self.regressor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dense",
                        nn.Linear(
                            in_features=self.recurrent_layers_sizes[1],
                            out_features=self.regressor_layer_size,
                        ),
                    ),
                    ("activation_relu", nn.ReLU()),
                    ("regressor_dropout", nn.Dropout(self.latent_dropout_rate)),
                ]
            )
        )

        self.output_layer = nn.Linear(
            in_features=self.regressor_layer_size, out_features=num_classes
        )

        self.activation = self.final_activation

        # start from the same weight distribution as the Keras model
        init_like_keras(self)

    def forward(self, inputs):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : tensor
            Input tensor (shape: [batch_size, seq_length]).

        Returns
        -------
        tensor
            Predicted output (shape: [batch_size, num_classes]).
        """
        x = self.embedding(inputs)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        x = self.activation(x)
        return x
