import tensorflow as tf

from ..constants import ALPHABET_UNMOD


@tf.keras.utils.register_keras_serializable(package="dlomix")
class DeepLCRetentionTimePredictor(tf.keras.Model):
    """
    DeepLC multi-branch CNN.

    Branches
    --------
    onehot_branch     : Conv on one-hot AA            → (batch_size, T', 2)  → Flatten
    aminoacid_branch  : Conv on per-pos atom counts   → (batch_size, T', 64) → Flatten
    diaminoacid_branch: Conv on di-AA atom counts     → (batch_size, T'', 64)→ Flatten
    global_branch     : Dense on peptide-level totals → (batch_size, 16)     (optional)

    All branches are concatenated, then passed through 5 X Dense(128, leaky_relu)
    and a final Dense(1) for the RT prediction.

        Input dict (all keys must be present in `inputs`):
            "seq"             : int32   (batch, MAX_LEN)     ← integer AA indices
                                                    OR float (batch, MAX_LEN, A)  ← precomputed one-hot
            "counts"          : float32 (batch, MAX_LEN, 6)  ← atoms_per_pos
            "di_counts"       : float32 (batch, MAX_LEN//2, 6)
            "global_features" : float32 (batch, 6)           ← only if use_global_features=True

    Usage
    -----
    model = DeepLCRetentionTimePredictor()
    preds = model(inputs)   # shape (batch, 1)

    # Build (to print summary / enable model.fit):
    model.build({
        "seq":       (None, 60),
        "counts":    (None, 60, 6),
        "di_counts": (None, 30, 6),
    })
    model.summary()
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

        leaky = lambda: tf.keras.layers.ReLU(max_value=20, negative_slope=0.1)

        # ── Branch: one-hot encoding ────────────────────────────────
        # Input: (batch, 60, 21)  Output: Flatten → (batch, ?)
        # Tanh + aggressive pooling (pool_size=10, strides=10) to
        # compress 60 positions
        self.onehot_branch = tf.keras.Sequential(
            [
                _build_conv_pool_block(
                    n_filters=2,
                    kernel=2,
                    padding="same",
                    activation="tanh",
                    pool=True,
                    pool_size=10,
                    pool_strides=10,
                ),
                tf.keras.layers.Flatten(),
            ],
            name="onehot_branch",
        )

        # ── Branch: per-position atom counts ────────────────────────
        # Input: (batch, 60, 6)   Output: Flatten → (batch, ?)
        self.aminoacid_branch = tf.keras.Sequential(
            [
                _build_conv_pool_block(n_filters=256, kernel=8, padding="same"),  # → /2
                _build_conv_pool_block(n_filters=128, kernel=8, padding="same"),  # → /4
                _build_conv_pool_block(
                    n_filters=64, kernel=8, padding="same", pool=False
                ),
                tf.keras.layers.Flatten(),
            ],
            name="aminoacid_branch",
        )

        # ── Branch: di-amino acid atom counts ───────────────────────
        # Input: (batch, 30, 6)   Output: Flatten → (batch, ?)
        self.diaminoacid_branch = tf.keras.Sequential(
            [
                _build_conv_pool_block(n_filters=128, kernel=2, padding="same"),  # → /2
                _build_conv_pool_block(n_filters=64, kernel=2, padding="same"),  # → /4
                tf.keras.layers.Flatten(),
            ],
            name="diaminoacid_branch",
        )

        # ── Branch: global features (optional) ──────────────────────
        # Input: (batch, 6)       Output: (batch, 16)
        if use_global_features:
            self.global_branch = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(16),
                    leaky(),
                    tf.keras.layers.Dense(16),
                    leaky(),
                    tf.keras.layers.Dense(16),
                    leaky(),
                ],
                name="global_branch",
            )

        # ── Regressor head ───────────────────────────────────────────
        # 5 × Dense(128, leaky_relu) → Dense(1)
        regressor_layers = []
        for _ in range(5):
            regressor_layers += [tf.keras.layers.Dense(128), leaky()]
        self.regressor = tf.keras.Sequential(regressor_layers, name="regressor")
        self.output_layer = tf.keras.layers.Dense(1, name="rt_output")

    def call(self, inputs: dict[str, tf.Tensor], training=False) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs : dict with keys "seq", "counts", "di_counts"
                 and optionally "global_features"

        Returns
        -------
        tf.Tensor of shape (batch, 1) – predicted retention time
        """
        # ── Ensure one-hot sequence representation ──────────────────
        # Accept either token ids: (batch, seq_len) or one-hot: (batch, seq_len, depth)
        seq = inputs[self.sequence_input_key]
        if seq.shape.rank == 2:
            one_hot = tf.one_hot(tf.cast(seq, tf.int32), depth=len(self.alphabet))
        elif seq.shape.rank == 3:
            if seq.shape[-1] is not None and seq.shape[-1] != len(self.alphabet):
                raise ValueError(
                    "`inputs['seq']` last dimension must match alphabet size "
                    f"({len(self.alphabet)}), got {seq.shape[-1]}."
                )
            one_hot = tf.cast(seq, tf.float32)
        else:
            raise ValueError(
                "`inputs['seq']` must have rank 2 (token ids) or rank 3 (one-hot), "
                f"got rank {seq.shape.rank}."
            )

        # ── Run branches ────────────────────────────────────────────
        onehot_output = self.onehot_branch(one_hot, training=training)  # (batch, flat)
        aminoacid_output = self.aminoacid_branch(
            inputs[self.counts_input_key], training=training
        )  # (batch, flat)
        diaminoacid_output = self.diaminoacid_branch(
            inputs[self.di_counts_input_key], training=training
        )  # (batch, flat)

        branch_outputs = [
            onehot_output,
            aminoacid_output,
            diaminoacid_output,
        ]

        if self.use_global_features:
            branch_outputs.append(
                self.global_branch(
                    inputs[self.global_features_input_key], training=training
                )
            )

        # ── Concatenate + regress ─────────────────────────────────
        x = tf.concat(branch_outputs, axis=1)
        x = self.regressor(x, training=training)
        return self.output_layer(x)

    def get_config(self):
        return {
            "seq_length": self.seq_length,
            "use_global_features": self.use_global_features,
            "alphabet": self.alphabet,
            "sequence_input_key": self.sequence_input_key,
            "counts_input_key": self.counts_input_key,
            "di_counts_input_key": self.di_counts_input_key,
            "global_features_input_key": self.global_features_input_key,
        }


def _build_conv_pool_block(
    n_conv_layers: int = 2,
    n_filters: int = 256,
    kernel: int = 8,
    padding: str = "same",
    activation: str = "leaky_relu",
    pool: bool = True,
    pool_size: int = 2,
    pool_strides: int = 2,
) -> tf.keras.Sequential:
    """
    Build a (Conv1D × n_conv_layers) + optional MaxPool block.

    """
    if activation == "leaky_relu":
        # LeakyReLU with a 20-unit cap and 0.1 negative slope
        act_fn = lambda: tf.keras.layers.ReLU(max_value=20, negative_slope=0.1)
    elif activation == "tanh":
        act_fn = lambda: tf.keras.layers.Activation("tanh")
    else:
        act_fn = lambda: tf.keras.layers.Activation("relu")

    layers_list = []
    for _ in range(n_conv_layers):
        layers_list.append(
            tf.keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=kernel,
                padding=padding,
                activation=None,
            )
        )
        layers_list.append(act_fn())

    if pool:
        layers_list.append(
            tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_strides)
        )

    return tf.keras.Sequential(layers_list)
