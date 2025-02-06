from collections import OrderedDict

import torch.nn as nn

from ..constants import ALPHABET_UNMOD
from ..data.processing.feature_extractors import FEATURE_EXTRACTORS_PARAMETERS
from ..layers.attention_torch import AttentionLayerTorch
from ..layers.bi_gru_seq_encoder_torch import BiGRUSequentialEncoder


class PrositRetentionTimePredictorTorch(nn.Module):
    """
    Implementation of the Prosit model for retention time prediction.

    Parameters
    -----------
    embedding_output_dim : int, optional
        Size of the embeddings to use. Defaults to 16.
    seq_length : int, optional
        Sequence length of the peptide sequences. Defaults to 30.
    alphabet : dict, optional
        Dictionary mapping for the alphabet (the amino acids in this case). Defaults to ALPHABET_UNMOD.
    dropout_rate : float, optional
        Probability to use for dropout layers in the encoder. Defaults to 0.5.
    latent_dropout_rate : float, optional
        Probability to use for dropout layers in the regressor layers after encoding. Defaults to 0.1.
    recurrent_layers_sizes : tuple, optional
        A tuple of 2 values for the sizes of the two GRU layers in the encoder. Defaults to (256, 512).
    regressor_layer_size : int, optional
        Size of the dense layer in the regressor after the encoder. Defaults to 512.


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
    ):
        super(PrositRetentionTimePredictorTorch, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(alphabet)

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes
        self.embedding_output_dim = embedding_output_dim
        self.seq_length = seq_length

        self.embedding = nn.Embedding(
            num_embeddings=self.embeddings_count,
            embedding_dim=embedding_output_dim,
            padding_idx=0,  # TODO check this
        )

        self.encoder = BiGRUSequentialEncoder(
            embedding_output_dim, self.recurrent_layers_sizes, self.dropout_rate
        )

        self.attention = AttentionLayerTorch(
            feature_dim=self.recurrent_layers_sizes[1], seq_len=self.seq_length
        )

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
            in_features=self.regressor_layer_size, out_features=1
        )

    def forward(self, inputs, **kwargs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        return x
