import torch.nn as nn

from .attention_torch import DecoderAttentionLayer


class GRUSequentialDecoder(nn.Module):
    """Decoder of the Prosit intensity model: a GRU followed by decoder attention.

    Mirrors the Keras decoder (``GRU`` -> ``Dropout`` -> ``DecoderAttentionLayer``).

    Args:
        input_size (int): Size of the features fed to the decoder, i.e. the size of the
            fused encoder output (``recurrent_layers_sizes[1]`` in the model).
        hidden_size (int): Number of units of the decoder GRU (``regressor_layer_size``
            in the model, as in the Keras implementation).
        dropout_rate (float): The dropout rate applied after the GRU.
        max_ion (int): Number of fragment ion positions the decoder attends over.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        dropout_rate,
        max_ion,
    ):
        super(GRUSequentialDecoder, self).__init__()
        self.unidirectional_GRU = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
        )
        self.encoder_dropout = nn.Dropout(dropout_rate)

        self.attention = DecoderAttentionLayer(max_ion)

    def forward(self, inputs):
        x, _ = self.unidirectional_GRU(inputs)
        x = self.encoder_dropout(x)
        x = self.attention(x)
        return x
