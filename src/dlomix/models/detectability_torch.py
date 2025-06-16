import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import CLASSES_LABELS, padding_char


class DetectabilityModel(nn.Module):
    def __init__(
        self,
        num_units,
        alphabet_size,
        num_classes=len(CLASSES_LABELS),
        padding_idx=padding_char,
    ):
        super(DetectabilityModel, self).__init__()

        self.num_units = num_units
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.alphabet_size = alphabet_size

        # Equivalent to Keras one-hot encoding layer
        self.embedding = nn.Embedding(
            alphabet_size, alphabet_size, padding_idx=padding_idx
        )

        self.encoder = Encoder(num_units, alphabet_size)
        self.decoder = Decoder(num_units, num_classes)

    def create_padding_mask(self, x):
        # Create mask where padding_idx tokens are 1 and others are 0
        mask = x == self.padding_idx
        return mask

    def forward(self, x):
        # Create padding mask
        padding_mask = self.create_padding_mask(x)

        # One-hot encoding
        x = self.embedding(x)

        # Encoder
        encoder_outputs, (state_f, state_b) = self.encoder(x, padding_mask)

        # Concatenate forward and backward states
        decoder_hidden = torch.cat([state_f, state_b], dim=-1)

        # Decoder
        output = self.decoder(
            decoder_hidden, state_f, state_b, encoder_outputs, padding_mask
        )

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
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=0.0
        )

        # Apply mask to output
        mask = mask.unsqueeze(-1).expand(-1, -1, output.size(-1))
        output = output.masked_fill(mask, 0.0)

        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()

        self.W1 = nn.LazyLinear(hidden_size)
        self.W2 = nn.LazyLinear(hidden_size)
        self.V = nn.LazyLinear(1)

    def forward(self, query, values, mask):
        # query shape: (batch_size, hidden_size)
        # values shape: (batch_size, seq_len, hidden_size)
        # mask shape: (batch_size, seq_len)

        # Add time axis to query
        query = query.unsqueeze(1)

        # Calculate attention scores
        query_values = torch.tanh(self.W1(query) + self.W2(values))
        scores = self.V(query_values)

        # Apply mask to attention scores
        # Set masked positions to -inf before softmax
        mask = mask.unsqueeze(-1)
        scores = scores.masked_fill(mask, float("-inf"))

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

    def forward(self, decoder_input, state_f, state_b, encoder_outputs, mask):
        # Apply attention
        context = self.attention(decoder_input, encoder_outputs, mask)

        # Add sequence dimension
        context = context.unsqueeze(1)

        # Pass through GRU
        states = torch.stack([state_f, state_b])
        output, hidden = self.gru(context, states)

        # Pass through final dense layer
        output = self.dense(output)
        output = F.softmax(output, dim=-1)

        return output.squeeze(1)
