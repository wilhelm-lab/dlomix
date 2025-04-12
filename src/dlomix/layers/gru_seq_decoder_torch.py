import torch.nn as nn

from .attention_torch import DecoderAttentionLayer


class GRUSequentialDecoder(nn.Module):
    """Encoder class needed to handle two GRU outputs in torch.

    No implementation using nn.Sequential() possible.

    Args:
        embedding_output_dim (int): The size of the embedding output dimension. Defaults to 16.
        recurrent_layers_sizes (tuple): The sizes of the recurrent layers. Defaults to (256, 512).
        dropout_rate (float): The dropout rate used in the encoder layers. Defaults to 0.5.
    """

    def __init__(
        self,
        recurrent_layers_sizes,
        dropout_rate,
        max_ion,
    ):
        super(GRUSequentialDecoder, self).__init__()
        self.unidirectional_GRU = nn.GRU(
            input_size=recurrent_layers_sizes[1],
            hidden_size=recurrent_layers_sizes[1],
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
