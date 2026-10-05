import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import ALPHABET_UNMOD
from ..layers.keras_initializers_torch import (
    KerasLazyConv1d,
    KerasLazyLinear,
    init_like_keras,
)


class DeepLCRetentionTimePredictor(nn.Module):
    """
    DeepLC multi-branch CNN (PyTorch implementation).

    Mirrors the TensorFlow implementation in :mod:`dlomix.models.deepLC` layer by
    layer, so that both compute the same function given the same weights.

    Branches
    --------
    onehot_branch     : Conv on one-hot AA            → (batch_size, T', 2)  → Flatten
    aminoacid_branch  : Conv on per-pos atom counts   → (batch_size, T', 64) → Flatten
    diaminoacid_branch: Conv on di-AA atom counts     → (batch_size, T'', 64)→ Flatten
    global_branch     : Dense on peptide-level totals → (batch_size, 16)     (optional)

    All branches are concatenated, then passed through 5 X Dense(128, leaky_relu)
    and a final Dense(1) for the RT prediction.

        Input dict (all keys must be present in `inputs`):
            "seq"             : int     (batch, MAX_LEN)     ← integer AA indices
                                                    OR float (batch, MAX_LEN, A)  ← precomputed one-hot
            "counts"          : float32 (batch, MAX_LEN, 6)  ← atoms_per_pos
            "di_counts"       : float32 (batch, MAX_LEN//2, 6)
            "global_features" : float32 (batch, F)           ← only if use_global_features=True

    The layers infer their input sizes on the first forward pass (as the Keras layers
    do when they are built), so run one batch through the model before counting or
    loading parameters. Weights are initialized as in Keras, see
    :mod:`dlomix.layers.keras_initializers_torch`.

    Usage
    -----
    model = DeepLCRetentionTimePredictor()
    preds = model(inputs)   # shape (batch, 1)
    """

    def __init__(
        self,
        seq_length: int = 60,
        use_global_features: bool = False,
        alphabet: dict = ALPHABET_UNMOD,
        sequence_input_key: str = "seq",
        counts_input_key: str = "counts",
        di_counts_input_key: str = "di_counts",
        global_features_input_key: str = "global_features",
    ):
        super().__init__()
        self.seq_length = seq_length
        self.use_global_features = use_global_features
        self.alphabet = alphabet
        self.sequence_input_key = sequence_input_key
        self.counts_input_key = counts_input_key
        self.di_counts_input_key = di_counts_input_key
        self.global_features_input_key = global_features_input_key

        # ── Branch: one-hot encoding ────────────────────────────────
        # Tanh + aggressive pooling (pool_size=10, strides=10) to
        # compress 60 positions
        self.onehot_branch = _ChannelsLastBranch(
            _build_conv_pool_block(
                n_filters=2,
                kernel=2,
                activation="tanh",
                pool=True,
                pool_size=10,
                pool_strides=10,
            )
        )

        # ── Branch: per-position atom counts ────────────────────────
        self.aminoacid_branch = _ChannelsLastBranch(
            _build_conv_pool_block(n_filters=256, kernel=8),  # → /2
            _build_conv_pool_block(n_filters=128, kernel=8),  # → /4
            _build_conv_pool_block(n_filters=64, kernel=8, pool=False),
        )

        # ── Branch: di-amino acid atom counts ───────────────────────
        self.diaminoacid_branch = _ChannelsLastBranch(
            _build_conv_pool_block(n_filters=128, kernel=2),  # → /2
            _build_conv_pool_block(n_filters=64, kernel=2),  # → /4
        )

        # ── Branch: global features (optional) ──────────────────────
        if use_global_features:
            self.global_branch = nn.Sequential(
                KerasLazyLinear(16),
                CappedLeakyReLU(),
                nn.Linear(16, 16),
                CappedLeakyReLU(),
                nn.Linear(16, 16),
                CappedLeakyReLU(),
            )

        # ── Regressor head ───────────────────────────────────────────
        # 5 × Dense(128, leaky_relu) → Dense(1)
        regressor_layers = [KerasLazyLinear(128), CappedLeakyReLU()]
        for _ in range(4):
            regressor_layers += [nn.Linear(128, 128), CappedLeakyReLU()]
        self.regressor = nn.Sequential(*regressor_layers)
        self.output_layer = nn.Linear(128, 1)

        # initialize like the Keras layers (lazy layers do so when they are created)
        init_like_keras(self)

    def forward(self, inputs: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : dict with keys "seq", "counts", "di_counts"
                 and optionally "global_features"

        Returns
        -------
        torch.Tensor of shape (batch, 1) – predicted retention time
        """
        # ── Ensure one-hot sequence representation ──────────────────
        # Accept either token ids: (batch, seq_len) or one-hot: (batch, seq_len, depth)
        seq = inputs[self.sequence_input_key]
        if seq.dim() == 2:
            one_hot = F.one_hot(seq.long(), num_classes=len(self.alphabet)).float()
        elif seq.dim() == 3:
            if seq.shape[-1] != len(self.alphabet):
                raise ValueError(
                    "`inputs['seq']` last dimension must match alphabet size "
                    f"({len(self.alphabet)}), got {seq.shape[-1]}."
                )
            one_hot = seq.float()
        else:
            raise ValueError(
                "`inputs['seq']` must have rank 2 (token ids) or rank 3 (one-hot), "
                f"got rank {seq.dim()}."
            )

        # ── Run branches ────────────────────────────────────────────
        branch_outputs = [
            self.onehot_branch(one_hot),
            self.aminoacid_branch(inputs[self.counts_input_key].float()),
            self.diaminoacid_branch(inputs[self.di_counts_input_key].float()),
        ]

        if self.use_global_features:
            branch_outputs.append(
                self.global_branch(inputs[self.global_features_input_key].float())
            )

        # ── Concatenate + regress ─────────────────────────────────
        x = torch.cat(branch_outputs, dim=1)
        x = self.regressor(x)
        return self.output_layer(x)


class CappedLeakyReLU(nn.Module):
    """``keras.layers.ReLU(max_value=20, negative_slope=0.1)``: leaky below 0, capped at 20."""

    def __init__(self, max_value: float = 20.0, negative_slope: float = 0.1):
        super().__init__()
        self.max_value = max_value
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.where(
            x < 0, self.negative_slope * x, torch.clamp(x, max=self.max_value)
        )


class _SamePad1d(nn.Module):
    """Pad the time axis the way Keras/TensorFlow ``padding="same"`` does.

    For an even kernel the extra step goes on the right (kernel 8 → 3 left, 4 right),
    which ``nn.Conv1d(padding="same")`` does not guarantee.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        total = kernel_size - 1
        self.left, self.right = total // 2, total - total // 2

    def forward(self, x):
        return F.pad(x, (self.left, self.right))


class _ChannelsLastBranch(nn.Module):
    """Run conv blocks on (batch, time, channels) input and flatten channels-last.

    PyTorch convolutions expect (batch, channels, time); flattening after moving the
    channels back last gives the same feature order as Keras' Flatten.
    """

    def __init__(self, *blocks: nn.Module):
        super().__init__()
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x.transpose(1, 2))
        return torch.flatten(x.transpose(1, 2), start_dim=1)


def _build_conv_pool_block(
    n_conv_layers: int = 2,
    n_filters: int = 256,
    kernel: int = 8,
    activation: str = "leaky_relu",
    pool: bool = True,
    pool_size: int = 2,
    pool_strides: int = 2,
) -> nn.Sequential:
    """
    Build a (Conv1D × n_conv_layers) + optional MaxPool block ("same" padding).

    """
    if activation == "leaky_relu":
        # LeakyReLU with a 20-unit cap and 0.1 negative slope
        act_fn = CappedLeakyReLU
    elif activation == "tanh":
        act_fn = nn.Tanh
    else:
        act_fn = nn.ReLU

    layers_list = []
    for _ in range(n_conv_layers):
        layers_list += [
            _SamePad1d(kernel),
            KerasLazyConv1d(out_channels=n_filters, kernel_size=kernel),
            act_fn(),
        ]

    if pool:
        layers_list.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_strides))

    return nn.Sequential(*layers_list)
