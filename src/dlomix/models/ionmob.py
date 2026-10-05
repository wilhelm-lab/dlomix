import keras
import numpy as np
import tensorflow as tf

from ..layers.gru_kernel import gru_kernel_kwargs
from ..losses.ionmob import MaskedIonmobLoss


@keras.saving.register_keras_serializable(package="dlomix")
class SquareRootProjectionLayer(tf.keras.layers.Layer):
    """Initial CCS estimate: a per-charge linear fit on the square root of m/z.

    Args:
        weights: initial slopes, one per charge state
        bias: initial intercepts, one per charge state
        trainable: whether the slopes and intercepts are optimized during training
    """

    def __init__(self, weights, bias, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.initial_slopes = [float(w) for w in np.ravel(weights)]
        self.initial_intercepts = [float(b) for b in np.ravel(bias)]
        self.slopes = self.add_weight(
            shape=(len(self.initial_slopes),),
            initializer=keras.initializers.Constant(self.initial_slopes),
            name="slopes",
        )
        self.intercepts = self.add_weight(
            shape=(len(self.initial_intercepts),),
            initializer=keras.initializers.Constant(self.initial_intercepts),
            name="intercepts",
        )

    def call(self, mz, charge):
        # mz: (batch, 1), charge: one-hot (batch, max_charge)
        projection = self.slopes * tf.sqrt(mz) + self.intercepts
        return tf.reduce_sum(projection * charge, axis=-1, keepdims=True)

    def get_config(self):
        config = super().get_config()
        config.update({"weights": self.initial_slopes, "bias": self.initial_intercepts})
        return config


@keras.saving.register_keras_serializable(package="dlomix")
class Ionmob(tf.keras.Model):
    """Ionmob model for CCS mean and standard deviation prediction (TensorFlow).

    Mirrors the PyTorch implementation in :mod:`dlomix.models.ionmob_torch` layer by
    layer, so that both compute the same function given the same weights, and returns
    the same outputs: ``(total_ccs, ccs_delta, ccs_std)``, where ``total_ccs`` is the
    square-root projection plus the learned correction ``ccs_delta``.

    Inputs are either a dict (as produced by :class:`~dlomix.data.IonMobilityDataset`)
    holding the tokenized sequence, the m/z and the integer charge state under
    ``sequence_key``, ``mz_key`` and ``charge_key``, or a ``(seq, mz, charge)``
    tuple, the argument order of the PyTorch model.

    Training with ``fit`` uses :class:`~dlomix.losses.MaskedIonmobLoss`, as in the
    PyTorch training loop: compile with ``loss=MaskedIonmobLoss()`` (the default when
    no loss is given). The targets are ``(ccs, ccs_std)``, either as a tuple or as a
    dict keyed by ``ccs_key`` and ``ccs_std_key`` (the dataset's label columns).

    Args:
        num_tokens: size of the token vocabulary
        initial_weights: initial fit weights for the square root projection layer
        initial_bias: initial fit bias(es) for the square root projection layer
        max_charge: highest charge state (charges are one-hot encoded)
        max_peptide_length: maximum peptide length (number of amino acids WITH modifications)
        emb_dim: embedding dimension size
        gru_1: size of the first GRU layer
        gru_2: size of the second GRU layer
        rdo: recurrent dropout rate. The PyTorch model passes it to a single-layer
            ``nn.GRU``, where it has no effect, so the two agree only at 0 (the default)
        do: dropout rate
    """

    def __init__(
        self,
        num_tokens,
        initial_weights=(12.3177, 15.0300, 17.1686, 21.1792),
        initial_bias=(-81.5547, 1.8667, 99.4165, 180.1543),
        max_charge: int = 4,
        max_peptide_length: int = 50,
        emb_dim: int = 64,
        gru_1: int = 64,
        gru_2: int = 32,
        rdo: float = 0.0,
        do: float = 0.2,
        sequence_key: str = "sequence_modified",
        mz_key: str = "mz",
        charge_key: str = "charge",
        ccs_key: str = "ccs",
        ccs_std_key: str = "ccs_std",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.initial_weights = [float(w) for w in np.ravel(initial_weights)]
        self.initial_bias = [float(b) for b in np.ravel(initial_bias)]
        self.max_charge = max_charge
        self.max_peptide_length = max_peptide_length
        self.emb_dim = emb_dim
        self.gru_1 = gru_1
        self.gru_2 = gru_2
        self.rdo = rdo
        self.do = do
        self.sequence_key = sequence_key
        self.mz_key = mz_key
        self.charge_key = charge_key
        self.ccs_key = ccs_key
        self.ccs_std_key = ccs_std_key

        self.initial = SquareRootProjectionLayer(
            self.initial_weights, self.initial_bias, trainable=True, name="initial"
        )
        self.emb = tf.keras.layers.Embedding(num_tokens, emb_dim, name="emb")
        self.gru1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                gru_1,
                return_sequences=True,
                recurrent_dropout=rdo,
                **gru_kernel_kwargs(),
            ),
            name="gru1",
        )
        # without return_sequences, the output is the last hidden state of both
        # directions, concatenated [forward, backward] (PyTorch's h_n)
        self.gru2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(gru_2, recurrent_dropout=rdo, **gru_kernel_kwargs()),
            name="gru2",
        )
        self.dropout = tf.keras.layers.Dropout(do)

        self.dense_ccs_1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense_ccs_2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense_ccs_std_1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense_ccs_std_2 = tf.keras.layers.Dense(64, activation="relu")
        self.out_ccs = tf.keras.layers.Dense(1)
        self.out_ccs_std = tf.keras.layers.Dense(1)

    def _unpack_inputs(self, inputs):
        if isinstance(inputs, dict):
            return (
                inputs[self.sequence_key],
                inputs[self.mz_key],
                inputs[self.charge_key],
            )
        seq, mz, charge = inputs
        return seq, mz, charge

    def call(self, inputs, training=False):
        """
        Args:
            inputs: dict with the sequence, m/z and charge, or a (seq, mz, charge) tuple

        Returns:
            total_output: initial sqrt prediction + deep learning prediction
            ccs_output: deep learning output for CCS (difference from initial prediction)
            ccs_std_output: CCS std prediction
        """
        seq, mz, charge = self._unpack_inputs(inputs)

        x_emb = self.emb(tf.cast(seq, tf.int32))

        # one-hot encode charge (the dataset may provide it as a float)
        charge = tf.one_hot(
            tf.cast(tf.reshape(charge, [-1]), tf.int32) - 1, depth=self.max_charge
        )

        mz = tf.cast(mz, tf.float32)
        if mz.shape.rank == 1:
            mz = tf.expand_dims(mz, axis=-1)

        x_recurrent = self.gru2(self.gru1(x_emb, training=training), training=training)

        # Concatenate charge and recurrent features
        concat = tf.concat([charge, x_recurrent], axis=-1)

        cc1 = self.dropout(self.dense_ccs_1(concat), training=training)
        cc2 = self.dense_ccs_2(cc1)

        ccs_std1 = self.dropout(self.dense_ccs_std_1(concat), training=training)
        ccs_std2 = self.dense_ccs_std_2(ccs_std1)

        initial_output = self.initial(mz, charge)
        ccs_output = self.out_ccs(cc2)
        ccs_std_output = self.out_ccs_std(ccs_std2)
        total_output = initial_output + ccs_output

        return total_output, ccs_output, ccs_std_output

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, training=True
    ):
        """Loss on (total_ccs, ccs_std) against the (ccs, ccs_std) targets.

        Keras pairs each output with one target, which does not fit a model with three
        outputs and two targets, so the loss is applied here as in the PyTorch loop:
        ``loss((total_ccs, ccs_std), (target_ccs, target_ccs_std))``.
        """
        loss_fn = self.loss if self.loss is not None else MaskedIonmobLoss()
        if isinstance(y, dict):
            targets = (y[self.ccs_key], y[self.ccs_std_key])
        else:
            targets = tuple(y)
        total_output, _, ccs_std_output = y_pred
        loss = loss_fn((total_output, ccs_std_output), targets)
        if self.losses:  # regularization losses, if any
            loss = loss + tf.add_n(self.losses)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_tokens": self.num_tokens,
                "initial_weights": self.initial_weights,
                "initial_bias": self.initial_bias,
                "max_charge": self.max_charge,
                "max_peptide_length": self.max_peptide_length,
                "emb_dim": self.emb_dim,
                "gru_1": self.gru_1,
                "gru_2": self.gru_2,
                "rdo": self.rdo,
                "do": self.do,
                "sequence_key": self.sequence_key,
                "mz_key": self.mz_key,
                "charge_key": self.charge_key,
                "ccs_key": self.ccs_key,
                "ccs_std_key": self.ccs_std_key,
            }
        )
        return config
