import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import CLASSES_LABELS, padding_char
from ..layers.keras_initializers_torch import KerasLazyLinear, init_like_keras


class DetectabilityModel(nn.Module):
    """Peptide detectability model (PyTorch); mirrors the TensorFlow implementation.

    Args:
        num_units: size of the GRU layers.
        alphabet_size: number of tokens; the sequence is one-hot encoded to this
            depth. Defaults to the size of the detectability alphabet.
        num_classes: number of detectability classes.
        padding_idx: token id of the padding, which is masked in the encoder.
    """

    def __init__(
        self,
        num_units,
        alphabet_size=len(padding_char),
        num_classes=len(CLASSES_LABELS),
        padding_idx=int(np.argmax(padding_char)),
    ):
        super(DetectabilityModel, self).__init__()

        self.num_units = num_units
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.alphabet_size = alphabet_size

        self.encoder = Encoder(num_units, alphabet_size)
        self.decoder = Decoder(num_units, num_classes)

        # start from the same weight distribution as the Keras model, whose GRUs use
        # a Glorot-uniform recurrent initializer
        init_like_keras(self, gru_recurrent_initializer="glorot_uniform")

    def create_padding_mask(self, x):
        # Create mask where padding_idx tokens are 1 and others are 0
        mask = x == self.padding_idx
        return mask

    def forward(self, x):
        # Create padding mask
        padding_mask = self.create_padding_mask(x)

        # one-hot encoding, as the Keras model does (not a trainable embedding)
        x = F.one_hot(x.long(), num_classes=self.alphabet_size).float()

        # Encoder
        encoder_outputs, (state_f, state_b) = self.encoder(x, padding_mask)

        # Concatenate forward and backward states
        decoder_hidden = torch.cat([state_f, state_b], dim=-1)

        # Decoder
        output = self.decoder(decoder_hidden, state_f, state_b, encoder_outputs)

        return output


class Encoder(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x, mask):
        # x shape: (batch_size, seq_len, input_size)
        # mask shape: (batch_size, seq_len)

        # Create packed sequence
        lengths = (~mask).sum(dim=1).cpu()  # Get lengths of non-padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # Process through GRU
        packed_output, hidden = self.gru(packed_x)

        # Unpack the sequence
        # total_length keeps the padded length even when every sequence of the batch
        # is shorter, so the output still lines up with the mask
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=0.0, total_length=x.size(1)
        )

        # Apply mask to output
        mask = mask.unsqueeze(-1).expand(-1, -1, output.size(-1))
        output = output.masked_fill(mask, 0.0)

        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()

        self.W1 = KerasLazyLinear(hidden_size)
        self.W2 = KerasLazyLinear(hidden_size)
        self.V = KerasLazyLinear(1)

    def forward(self, query, values):
        # query shape: (batch_size, hidden_size)
        # values shape: (batch_size, seq_len, hidden_size)

        # As in the Keras model, padded positions are not masked here: their encoder
        # outputs are zero, so they add nothing to the context but still take part in
        # the softmax. Masking them would change the model the pretrained Keras
        # weights were trained as.

        # Add time axis to query
        query = query.unsqueeze(1)

        # Calculate attention scores
        query_values = torch.tanh(self.W1(query) + self.W2(values))
        scores = self.V(query_values)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)

        # Apply attention weights to values
        context = attention_weights * values

        # Sum over the time axis
        context = torch.sum(context, dim=1)

        return context


class Decoder(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.attention = BahdanauAttention(hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )

        self.dense = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, decoder_input, state_f, state_b, encoder_outputs):
        # Apply attention
        context = self.attention(decoder_input, encoder_outputs)

        # Add sequence dimension
        context = context.unsqueeze(1)

        # Pass through GRU
        states = torch.stack([state_f, state_b])
        output, hidden = self.gru(context, states)

        # Pass through final dense layer
        output = self.dense(output)
        output = F.softmax(output, dim=-1)

        return output.squeeze(1)
