import numpy as np
import torch
import torch.nn as nn


class SquareRootProjectionLayer(nn.Module):
    def __init__(self, weights, bias, trainable=True):
        """
        Initialize the SquareRootProjectionLayer.
        Args:
            weights: array of weights
            bias: array of biases
            trainable: whether the weights and biases are optimized during training
        """
        super(SquareRootProjectionLayer, self).__init__()
        self.trainable = trainable
        weights = torch.tensor(weights, dtype=torch.float32)
        bias = torch.tensor(bias, dtype=torch.float32)
        if self.trainable:
            self.slopes = nn.Parameter(weights)
            self.intercepts = nn.Parameter(bias)
        else:
            self.register_buffer("weights", weights)
            self.register_buffer("bias", bias)

    def forward(self, mz, charge):
        sqrt_mz = torch.sqrt(mz)
        sqrt_mz = sqrt_mz.expand(-1, self.slopes.shape[0])
        projection = (self.slopes * sqrt_mz) + self.intercepts
        projection = projection * charge
        result = torch.sum(projection, dim=-1, keepdim=True)
        return result


class Ionmob(nn.Module):
    def __init__(
        self,
        num_tokens,
        initial_weights: np.ndarray = np.array([12.3177, 15.0300, 17.1686, 21.1792]),
        initial_bias: np.ndarray = np.array([-81.5547, 1.8667, 99.4165, 180.1543]),
        max_charge: int = 4,
        max_peptide_length: int = 50,
        emb_dim: int = 64,
        gru_1: int = 64,
        gru_2: int = 32,
        rdo: float = 0.0,
        do: float = 0.2,
    ):
        """
        Initialize the Ionmob model for CCS mean and std prediction.
        Args:
            initial_weights: initial fit weights for the square root projection layer
            initial_bias: initial fit bias(es) for the square root projection layer
            num_tokens: size of the token vocabulary
            max_peptide_length: maximum peptide length (number of amino acids WITH modifications)
            emb_dim: embedding dimension size
            gru_1: size of the first GRU layer
            gru_2: size of the second GRU layer
            rdo: recurrent dropout rate
            do: dropout rate
        """
        super(Ionmob, self).__init__()
        self.max_charge = max_charge
        self.max_peptide_length = max_peptide_length
        self.initial = SquareRootProjectionLayer(
            initial_weights, initial_bias, trainable=True
        )
        self.emb = nn.Embedding(num_tokens, emb_dim)
        self.gru1 = nn.GRU(
            emb_dim, gru_1, batch_first=True, bidirectional=True, dropout=rdo
        )
        self.gru2 = nn.GRU(
            gru_1 * 2, gru_2, batch_first=True, bidirectional=True, dropout=rdo
        )
        self.dropout = nn.Dropout(do)

        # The dense layer input size is the size of the
        # second GRU layer * 2 (bidirectional) + max_charge
        dense1_input_size = gru_2 * 2 + max_charge
        self.dense_ccs_1 = nn.Linear(dense1_input_size, 128)
        self.dense_ccs_2 = nn.Linear(128, 64)

        self.dense_ccs_std_1 = nn.Linear(dense1_input_size, 128)
        self.dense_ccs_std_2 = nn.Linear(128, 64)

        self.out_ccs = nn.Linear(64, 1)
        self.out_ccs_std = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, seq, mz, charge):
        """
        Forward pass of the Ionmob model.
        Args:
            seq: Tokenized peptide sequence
            mz: Mass-over-charge (monoisotopic)
            charge: Charge state of the ion

        Returns:
            total_output: initial sqrt prediction + deep learning prediction
            ccs_output: deep learning output for CCS (difference from initial prediction)
            ccs_std_output: CCS std prediction
        """

        # Ensure seq is of type torch.long
        seq = seq.long()
        batch_size = mz.size(0)
        x_emb = self.emb(seq)

        # one-hot encode charge
        charge = torch.nn.functional.one_hot(
            charge - 1, num_classes=self.max_charge
        ).float()

        # check if mz is (batch_, 1) and not (batch_,), otherwise expand
        if mz.dim() == 1:
            mz = mz.unsqueeze(1)

        x_gru1, _ = self.gru1(x_emb)
        x_gru2, h_n = self.gru2(x_gru1)

        # Get the last hidden state from both directions
        x_recurrent = h_n.permute(1, 0, 2).reshape(batch_size, -1)

        # Concatenate charge and recurrent features
        concat = torch.cat([charge, x_recurrent], dim=-1)

        # Dense layers with activation and dropout
        cc1 = self.dropout(self.relu(self.dense_ccs_1(concat)))
        cc2 = self.relu(self.dense_ccs_2(cc1))

        ccs_std1 = self.dropout(self.relu(self.dense_ccs_std_1(concat)))
        ccs_std2 = self.relu(self.dense_ccs_std_2(ccs_std1))

        # Outputs
        initial_output = self.initial(mz, charge)
        ccs_output = self.out_ccs(cc2)

        ccs_std_output = self.out_ccs_std(ccs_std2)
        total_output = initial_output + ccs_output

        return total_output, ccs_output, ccs_std_output
