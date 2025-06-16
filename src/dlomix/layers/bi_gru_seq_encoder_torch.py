import torch.nn as nn


class BiGRUSequentialEncoder(nn.Module):
    """Encoder class needed to handle two GRU outputs in torch.

    No implementation using nn.Sequential() possible.

    Args:
        embedding_output_dim (int): The size of the embedding output dimension. Defaults to 16.
        recurrent_layers_sizes (tuple): The sizes of the recurrent layers. Defaults to (256, 512).
        dropout_rate (float): The dropout rate used in the encoder layers. Defaults to 0.5.
    """

    def __init__(
        self,
        embedding_output_dim,
        recurrent_layers_sizes,
        dropout_rate,
    ):
        super(BiGRUSequentialEncoder, self).__init__()

        self.bidirectional_GRU = nn.GRU(
            input_size=embedding_output_dim,
            hidden_size=recurrent_layers_sizes[0],
            batch_first=True,
            bidirectional=True,
        )
        self.encoder_dropout1 = nn.Dropout(dropout_rate)
        self.unidirectional_GRU = nn.GRU(
            input_size=recurrent_layers_sizes[0] * 2,
            hidden_size=recurrent_layers_sizes[1],
            batch_first=True,
            bidirectional=False,
        )
        self.encoder_dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x, _ = self.bidirectional_GRU(inputs)
        x = self.encoder_dropout1(x)
        x, _ = self.unidirectional_GRU(x)
        x = self.encoder_dropout2(x)
        return x
