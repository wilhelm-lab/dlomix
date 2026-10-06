"""The TensorFlow and PyTorch implementations of a model compute the same function.

Each test builds both implementations, copies the Keras weights into the PyTorch
model and requires the same outputs for the same inputs. Shape-only comparisons
would miss, for example, a different activation or padding.
"""

import keras
import numpy as np
import pytest
import torch
from datasets import Dataset

from dlomix.losses.ionmob import MaskedIonmobLoss
from dlomix.models.deepLC_torch import (
    DeepLCRetentionTimePredictor as DeepLCRetentionTimePredictorTorch,
)
from dlomix.models.ionmob_torch import Ionmob as IonmobTorch

# tf.keras *is* Keras 3, whose backend is one process-wide setting, so the TensorFlow
# models only build TensorFlow layers while Keras runs on TensorFlow (the default
# DLOMIX_BACKEND). The PyTorch-only tests below run under either backend.
requires_tensorflow_keras_backend = pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason=(
        "compares a tf.keras model against a torch model in one process, which "
        f"needs the TensorFlow Keras backend (currently '{keras.backend.backend()}')"
    ),
)

RTOL, ATOL = 1e-4, 1e-5


def _to_torch(array):
    return torch.from_numpy(np.ascontiguousarray(array))


def _keras_gates_to_torch(matrix):
    """Keras orders the GRU gates [z, r, h], PyTorch [r, z, n]."""
    z, r, h = np.split(matrix, 3, axis=-1)
    return np.concatenate([r, z, h], axis=-1)


def _keras_gru_to_torch_state(cell, prefix, suffix=""):
    return {
        f"{prefix}.weight_ih_l0{suffix}": _keras_gates_to_torch(cell.kernel.numpy()).T,
        f"{prefix}.weight_hh_l0{suffix}": _keras_gates_to_torch(
            cell.recurrent_kernel.numpy()
        ).T,
        f"{prefix}.bias_ih_l0{suffix}": _keras_gates_to_torch(cell.bias.numpy()[0]),
        f"{prefix}.bias_hh_l0{suffix}": _keras_gates_to_torch(cell.bias.numpy()[1]),
    }


def _load_state(torch_model, state):
    own = torch_model.state_dict()
    assert set(state) == set(own), (
        f"missing {sorted(set(own) - set(state))}, "
        f"unexpected {sorted(set(state) - set(own))}"
    )
    for name, value in state.items():
        if isinstance(own[name], torch.nn.parameter.UninitializedParameter):
            continue  # sized on the first forward pass, or here by load_state_dict
        assert tuple(own[name].shape) == value.shape, name
    torch_model.load_state_dict({k: _to_torch(v) for k, v in state.items()})


def _keras_layers_of_type(layer, layer_type):
    """All sublayers of ``layer_type``, depth first, in construction order."""
    found = []
    for sublayer in getattr(layer, "layers", []):
        if isinstance(sublayer, layer_type):
            found.append(sublayer)
        found.extend(_keras_layers_of_type(sublayer, layer_type))
    return found


# --------------------------------- DeepLC -------------------------------------


def _deeplc_inputs(use_global_features, batch_size=4, seq_length=60, n_global=55):
    # magnitudes as in real DeepLC features: atom counts up to ~30 per residue, and
    # peptide totals in the hundreds, so activations also reach the cap of 20
    rng = np.random.default_rng(0)
    inputs = {
        "seq": rng.integers(0, 20, size=(batch_size, seq_length)),
        "counts": rng.integers(0, 30, size=(batch_size, seq_length, 6)).astype(
            np.float32
        ),
        "di_counts": rng.integers(0, 60, size=(batch_size, seq_length // 2, 6)).astype(
            np.float32
        ),
    }
    if use_global_features:
        inputs["global_features"] = rng.uniform(
            0, 300, size=(batch_size, n_global)
        ).astype(np.float32)
    return inputs


def _deeplc_state(model_tf, model_torch):
    state = {}
    branches = ["onehot_branch", "aminoacid_branch", "diaminoacid_branch"]
    if model_tf.use_global_features:
        branches.append("global_branch")

    for branch in branches:
        keras_branch = getattr(model_tf, branch)
        torch_branch = getattr(model_torch, branch)
        keras_convs = _keras_layers_of_type(keras_branch, keras.layers.Conv1D)
        keras_denses = _keras_layers_of_type(keras_branch, keras.layers.Dense)
        torch_params = [
            (name, module)
            for name, module in torch_branch.named_modules()
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear))
        ]
        assert len(torch_params) == len(keras_convs) + len(keras_denses), branch
        for (name, module), keras_layer in zip(
            torch_params, keras_convs + keras_denses
        ):
            kernel = keras_layer.kernel.numpy()
            if isinstance(module, torch.nn.Conv1d):
                # Keras (kernel, in, out) -> PyTorch (out, in, kernel)
                state[f"{branch}.{name}.weight"] = kernel.transpose(2, 1, 0)
            else:
                state[f"{branch}.{name}.weight"] = kernel.T
            state[f"{branch}.{name}.bias"] = keras_layer.bias.numpy()

    keras_regressor = _keras_layers_of_type(model_tf.regressor, keras.layers.Dense)
    torch_regressor = [
        name
        for name, module in model_torch.regressor.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    for name, keras_layer in zip(torch_regressor, keras_regressor):
        state[f"regressor.{name}.weight"] = keras_layer.kernel.numpy().T
        state[f"regressor.{name}.bias"] = keras_layer.bias.numpy()
    state["output_layer.weight"] = model_tf.output_layer.kernel.numpy().T
    state["output_layer.bias"] = model_tf.output_layer.bias.numpy()
    return state


@requires_tensorflow_keras_backend
@pytest.mark.parametrize("use_global_features", [False, True])
def test_deeplc_tf_torch_same_function(use_global_features):
    from dlomix.models.deepLC import DeepLCRetentionTimePredictor

    inputs = _deeplc_inputs(use_global_features)
    model_tf = DeepLCRetentionTimePredictor(use_global_features=use_global_features)
    model_torch = DeepLCRetentionTimePredictorTorch(
        use_global_features=use_global_features
    )

    inputs_torch = {k: _to_torch(v) for k, v in inputs.items()}
    output_tf = np.asarray(model_tf(inputs))
    model_torch(inputs_torch)  # creates the lazy layers
    _load_state(model_torch, _deeplc_state(model_tf, model_torch))

    assert model_tf.count_params() == sum(p.numel() for p in model_torch.parameters())

    model_torch.eval()
    with torch.no_grad():
        output_torch = model_torch(inputs_torch)
    np.testing.assert_allclose(output_torch.numpy(), output_tf, rtol=RTOL, atol=ATOL)


@requires_tensorflow_keras_backend
def test_deeplc_tf_torch_same_function_with_one_hot_sequence():
    from dlomix.models.deepLC import DeepLCRetentionTimePredictor

    inputs = _deeplc_inputs(use_global_features=False)
    model_tf = DeepLCRetentionTimePredictor()
    model_torch = DeepLCRetentionTimePredictorTorch()
    output_ids = np.asarray(model_tf(inputs))

    one_hot = np.eye(len(model_tf.alphabet), dtype=np.float32)[inputs["seq"]]
    inputs_one_hot = {k: _to_torch(v) for k, v in {**inputs, "seq": one_hot}.items()}
    model_torch(inputs_one_hot)  # creates the lazy layers
    _load_state(model_torch, _deeplc_state(model_tf, model_torch))

    with torch.no_grad():
        output_torch = model_torch(inputs_one_hot)
    np.testing.assert_allclose(output_torch.numpy(), output_ids, rtol=RTOL, atol=ATOL)


def test_deeplc_torch_trains():
    torch.manual_seed(0)
    inputs = {k: _to_torch(v) for k, v in _deeplc_inputs(True).items()}
    model = DeepLCRetentionTimePredictorTorch(use_global_features=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # one unit away from the initial prediction: a fixed target can lie close to a
    # random initialization, leaving nothing to learn (the loss then went up in CI)
    target = model(inputs).detach() + 1.0

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = torch.nn.functional.l1_loss(model(inputs), target)
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_deeplc_torch_rejects_invalid_sequence_input():
    model = DeepLCRetentionTimePredictorTorch()
    inputs = {k: _to_torch(v) for k, v in _deeplc_inputs(False).items()}
    with pytest.raises(ValueError, match="rank 2"):
        model({**inputs, "seq": torch.zeros(4)})
    with pytest.raises(ValueError, match="alphabet size"):
        model({**inputs, "seq": torch.zeros(4, 60, 3)})


# --------------------------------- Ionmob -------------------------------------


NUM_TOKENS = 30


def _ionmob_inputs(batch_size=6, seq_length=20):
    """(seq, mz, charge), the positional inputs of both Ionmob implementations."""
    rng = np.random.default_rng(1)
    return (
        rng.integers(0, NUM_TOKENS, size=(batch_size, seq_length)),
        rng.uniform(400, 1500, size=(batch_size, 1)).astype(np.float32),
        rng.integers(1, 5, size=(batch_size,)),
    )


def _ionmob_state(model_tf):
    state = {
        "emb.weight": model_tf.emb.embeddings.numpy(),
        "initial.slopes": model_tf.initial.slopes.numpy(),
        "initial.intercepts": model_tf.initial.intercepts.numpy(),
    }
    for name in ("gru1", "gru2"):
        bidirectional = getattr(model_tf, name)
        state.update(_keras_gru_to_torch_state(bidirectional.forward_layer.cell, name))
        state.update(
            _keras_gru_to_torch_state(
                bidirectional.backward_layer.cell, name, "_reverse"
            )
        )
    for name in (
        "dense_ccs_1",
        "dense_ccs_2",
        "dense_ccs_std_1",
        "dense_ccs_std_2",
        "out_ccs",
        "out_ccs_std",
    ):
        state[f"{name}.weight"] = getattr(model_tf, name).kernel.numpy().T
        state[f"{name}.bias"] = getattr(model_tf, name).bias.numpy()
    return state


@requires_tensorflow_keras_backend
def test_ionmob_tf_torch_same_function():
    from dlomix.models.ionmob import Ionmob

    seq, mz, charge = _ionmob_inputs()
    inputs_dict = {"sequence_modified": seq, "mz": mz, "charge": charge}
    model_tf = Ionmob(num_tokens=NUM_TOKENS)
    model_torch = IonmobTorch(num_tokens=NUM_TOKENS)

    model_tf(inputs_dict)  # builds the layers
    # give every weight, including the GRU biases (zero at init), a non-trivial value
    rng = np.random.default_rng(2)
    for variable in model_tf.weights:
        variable.assign(
            variable.numpy()
            + rng.normal(scale=0.1, size=variable.shape).astype("float32")
        )
    _load_state(model_torch, _ionmob_state(model_tf))
    assert model_tf.count_params() == sum(p.numel() for p in model_torch.parameters())

    # both input forms of the Keras model, against the PyTorch call
    outputs_dict = model_tf(inputs_dict)
    outputs_tuple = model_tf((seq, mz, charge))
    model_torch.eval()
    with torch.no_grad():
        outputs_torch = model_torch(_to_torch(seq), _to_torch(mz), _to_torch(charge))

    for out_dict, out_tuple, out_torch in zip(
        outputs_dict, outputs_tuple, outputs_torch
    ):
        np.testing.assert_allclose(np.asarray(out_dict), np.asarray(out_tuple))
        np.testing.assert_allclose(
            out_torch.numpy(), np.asarray(out_dict), rtol=RTOL, atol=1e-3
        )


@requires_tensorflow_keras_backend
def test_ionmob_tf_trains_on_ionmobility_dataset():
    from dlomix.data import IonMobilityDataset
    from dlomix.models.ionmob import Ionmob

    rng = np.random.default_rng(4)
    n = 32
    data = Dataset.from_dict(
        {
            "sequence_modified": ["ACDEK", "PEPTIDEK", "MKLVAAR", "GGSSK"] * (n // 4),
            "ccs": rng.uniform(300, 500, n).tolist(),
            "ccs_std": np.where(rng.random(n) < 0.3, -1.0, 2.0).tolist(),
            "charge": rng.integers(1, 5, n).tolist(),
            "mz": rng.uniform(400, 1500, n).tolist(),
        }
    )
    dataset = IonMobilityDataset(
        data_format="hf",
        data_source=data,
        val_ratio=0.25,
        max_seq_len=10,
        batch_size=8,
        dataset_type="tf",
    )
    model = Ionmob(num_tokens=len(dataset.extended_alphabet))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2), loss=MaskedIonmobLoss(use_mse=True)
    )
    history = model.fit(
        dataset.tensor_train_data,
        validation_data=dataset.tensor_val_data,
        epochs=3,
        verbose=0,
    )
    assert np.all(np.isfinite(history.history["loss"]))
    assert history.history["loss"][-1] < history.history["loss"][0]

    total, delta, std = model.predict(dataset.tensor_val_data, verbose=0)
    assert total.shape == delta.shape == std.shape == (len(dataset["val"]), 1)


@requires_tensorflow_keras_backend
def test_ionmob_tf_config_round_trip():
    from dlomix.models.ionmob import Ionmob

    model = Ionmob(num_tokens=NUM_TOKENS, gru_1=16, gru_2=8, max_charge=4)
    clone = Ionmob.from_config(model.get_config())
    inputs = _ionmob_inputs()
    model(inputs)
    clone(inputs)
    assert clone.count_params() == model.count_params()
    np.testing.assert_allclose(
        clone.initial.slopes.numpy(), model.initial.slopes.numpy()
    )


# ------------------------ Prosit RT, charge state, detectability -------------------

# a vocabulary with the padding token at 0, as the datasets produce it
ALPHABET = {token: index for index, token in enumerate("-XACDEFGHIKLMNPQRSTVWY")}


def _padded_sequences(vocab_size, batch_size=4, seq_length=30, seed=5):
    """Token ids padded with 0 at the end, as the datasets produce them."""
    rng = np.random.default_rng(seed)
    sequences = np.zeros((batch_size, seq_length), dtype=np.int64)
    for row, length in enumerate(rng.integers(7, seq_length, size=batch_size)):
        sequences[row, :length] = rng.integers(1, vocab_size, size=length)
    return sequences


def _prepare_keras_model(model_tf, inputs):
    """Build the model and make every weight non-trivial.

    Keras initializes biases to zero, so without the perturbation a mapping or
    implementation error in a bias would go unnoticed. (On a Mac with an Apple GPU,
    these tests also check dlomix.layers.gru_kernel: the fused Metal GRU kernel
    would give other outputs with these non-zero biases.)
    """
    model_tf(inputs)
    rng = np.random.default_rng(6)
    for variable in model_tf.weights:
        variable.assign(
            variable.numpy()
            + rng.normal(scale=0.1, size=variable.shape).astype("float32")
        )


def _encoder_state(model_tf, torch_prefix="encoder"):
    """The Prosit-style encoder (BiGRU -> GRU) and attention, as a PyTorch state."""
    bidirectional, gru = model_tf.encoder.layers[0], model_tf.encoder.layers[2]
    state = {
        "embedding.weight": model_tf.embedding.embeddings.numpy(),
        "attention.W": model_tf.attention.W.numpy(),
        "attention.b": model_tf.attention.b.numpy(),
    }
    state.update(
        _keras_gru_to_torch_state(
            bidirectional.forward_layer.cell, f"{torch_prefix}.bidirectional_GRU"
        )
    )
    state.update(
        _keras_gru_to_torch_state(
            bidirectional.backward_layer.cell,
            f"{torch_prefix}.bidirectional_GRU",
            "_reverse",
        )
    )
    state.update(
        _keras_gru_to_torch_state(gru.cell, f"{torch_prefix}.unidirectional_GRU")
    )
    return state


def _regressor_state(model_tf):
    dense = model_tf.regressor.layers[0]
    return {
        "regressor.dense.weight": dense.kernel.numpy().T,
        "regressor.dense.bias": dense.bias.numpy(),
        "output_layer.weight": model_tf.output_layer.kernel.numpy().T,
        "output_layer.bias": model_tf.output_layer.bias.numpy(),
    }


@requires_tensorflow_keras_backend
def test_prosit_rt_tf_torch_same_function():
    from dlomix.models.prosit import PrositRetentionTimePredictor
    from dlomix.models.prosit_torch import (
        PrositRetentionTimePredictor as PrositRetentionTimePredictorTorch,
    )

    sequences = _padded_sequences(len(ALPHABET))
    model_tf = PrositRetentionTimePredictor(seq_length=30, alphabet=ALPHABET)
    model_torch = PrositRetentionTimePredictorTorch(seq_length=30, alphabet=ALPHABET)
    _prepare_keras_model(model_tf, sequences)
    _load_state(model_torch, {**_encoder_state(model_tf), **_regressor_state(model_tf)})

    model_torch.eval()
    with torch.no_grad():
        output_torch = model_torch(_to_torch(sequences))
    np.testing.assert_allclose(
        output_torch.numpy(), np.asarray(model_tf(sequences)), rtol=RTOL, atol=ATOL
    )


@requires_tensorflow_keras_backend
@pytest.mark.parametrize("flavour", ["relative", "observed", "dominant"])
def test_chargestate_tf_torch_same_function(flavour):
    from dlomix.models.chargestate import ChargeStatePredictor
    from dlomix.models.chargestate_torch import (
        ChargeStatePredictor as ChargeStatePredictorTorch,
    )

    sequences = _padded_sequences(len(ALPHABET))
    kwargs = dict(seq_length=30, alphabet=ALPHABET, model_flavour=flavour)
    model_tf = ChargeStatePredictor(**kwargs)
    model_torch = ChargeStatePredictorTorch(**kwargs)
    _prepare_keras_model(model_tf, sequences)
    _load_state(model_torch, {**_encoder_state(model_tf), **_regressor_state(model_tf)})

    model_torch.eval()
    with torch.no_grad():
        output_torch = model_torch(_to_torch(sequences))
    np.testing.assert_allclose(
        output_torch.numpy(), np.asarray(model_tf(sequences)), rtol=RTOL, atol=ATOL
    )


def _detectability_state(model_tf):
    encoder = model_tf.encoder.encoder_bi
    decoder = model_tf.decoder
    state = {}
    for prefix, bidirectional in (
        ("encoder.gru", encoder),
        ("decoder.gru", decoder.decoder_bi),
    ):
        state.update(
            _keras_gru_to_torch_state(bidirectional.forward_layer.cell, prefix)
        )
        state.update(
            _keras_gru_to_torch_state(
                bidirectional.backward_layer.cell, prefix, "_reverse"
            )
        )
    for name in ("W1", "W2", "V"):
        dense = getattr(decoder.attention, name)
        state[f"decoder.attention.{name}.weight"] = dense.kernel.numpy().T
        state[f"decoder.attention.{name}.bias"] = dense.bias.numpy()
    state["decoder.dense.weight"] = decoder.decoder_dense.kernel.numpy().T
    state["decoder.dense.bias"] = decoder.decoder_dense.bias.numpy()
    return state


@requires_tensorflow_keras_backend
def test_detectability_tf_torch_same_function():
    from dlomix.constants import alphabet
    from dlomix.models.detectability import DetectabilityModel
    from dlomix.models.detectability_torch import (
        DetectabilityModel as DetectabilityModelTorch,
    )

    sequences = _padded_sequences(len(alphabet), seq_length=40)
    model_tf = DetectabilityModel(num_units=16)
    model_torch = DetectabilityModelTorch(num_units=16, alphabet_size=len(alphabet))
    _prepare_keras_model(model_tf, sequences)
    model_torch(_to_torch(sequences))  # creates the lazy layers
    _load_state(model_torch, _detectability_state(model_tf))

    model_torch.eval()
    with torch.no_grad():
        output_torch = model_torch(_to_torch(sequences))
    np.testing.assert_allclose(
        output_torch.numpy(), np.asarray(model_tf(sequences)), rtol=RTOL, atol=ATOL
    )


def _intensity_state(model_tf):
    """The Prosit intensity model with meta data, as a PyTorch state."""
    bidirectional = model_tf.sequence_encoder.layers[0]
    meta_dense = model_tf.meta_encoder.layers[1]
    decoder_attention = model_tf.decoder.layers[2].dense
    time_dense = model_tf.regressor.layers[0].layer
    state = {
        "embedding.weight": model_tf.embedding.embeddings.numpy(),
        "attention.W": model_tf.attention.W.numpy(),
        "attention.b": model_tf.attention.b.numpy(),
        "meta_encoder.meta_dense.weight": meta_dense.kernel.numpy().T,
        "meta_encoder.meta_dense.bias": meta_dense.bias.numpy(),
        "decoder.attention.linear.weight": decoder_attention.kernel.numpy().T,
        "decoder.attention.linear.bias": decoder_attention.bias.numpy(),
        "regressor.time_dense.weight": time_dense.kernel.numpy().T,
        "regressor.time_dense.bias": time_dense.bias.numpy(),
    }
    for keras_layer, prefix, suffix in (
        (bidirectional.forward_layer, "sequence_encoder.bidirectional_GRU", ""),
        (
            bidirectional.backward_layer,
            "sequence_encoder.bidirectional_GRU",
            "_reverse",
        ),
        (
            model_tf.sequence_encoder.layers[2],
            "sequence_encoder.unidirectional_GRU",
            "",
        ),
        (model_tf.decoder.layers[0], "decoder.unidirectional_GRU", ""),
    ):
        state.update(_keras_gru_to_torch_state(keras_layer.cell, prefix, suffix))
    return state


@requires_tensorflow_keras_backend
@pytest.mark.parametrize("with_termini", [False, True])
def test_prosit_intensity_tf_torch_same_function(with_termini):
    from dlomix.models.prosit import PrositIntensityPredictor
    from dlomix.models.prosit_torch import (
        PrositIntensityPredictor as PrositIntensityPredictorTorch,
    )

    width = 32 if with_termini else 30
    rng = np.random.default_rng(7)
    inputs = {
        "sequence": _padded_sequences(len(ALPHABET), seq_length=width),
        "charge": np.eye(6, dtype="float32")[rng.integers(0, 6, size=4)],
        "ce": rng.uniform(0.2, 0.4, size=(4, 1)).astype("float32"),
    }
    kwargs = dict(
        seq_length=30,
        with_termini=with_termini,
        alphabet=ALPHABET,
        use_meta_data=True,
        input_keys={"SEQUENCE_KEY": "sequence"},
        meta_data_keys={"COLLISION_ENERGY_KEY": "ce", "PRECURSOR_CHARGE_KEY": "charge"},
    )
    model_tf = PrositIntensityPredictor(**kwargs)
    model_torch = PrositIntensityPredictorTorch(**kwargs)
    _prepare_keras_model(model_tf, inputs)
    inputs_torch = {k: _to_torch(v) for k, v in inputs.items()}
    model_torch(inputs_torch)  # creates the lazy layers
    _load_state(model_torch, _intensity_state(model_tf))

    model_torch.eval()
    with torch.no_grad():
        output_torch = model_torch(inputs_torch)
    np.testing.assert_allclose(
        output_torch.numpy(), np.asarray(model_tf(inputs)), rtol=RTOL, atol=ATOL
    )


def _assert_same_initial_distribution(keras_state, model_torch):
    """Each PyTorch parameter starts from the distribution of its Keras counterpart.

    Zero-initialized Keras weights (biases) must be zero in PyTorch too; for the
    others the standard deviations must agree (a different initializer changes
    them by a factor of 1.5 or more, sampling noise by a few percent).
    """
    torch_state = model_torch.state_dict()
    for name, keras_value in keras_state.items():
        torch_value = torch_state[name].numpy()
        if not np.any(keras_value):
            assert not np.any(torch_value), f"{name} is not zero-initialized"
        elif keras_value.size >= 64:
            np.testing.assert_allclose(
                torch_value.std(), keras_value.std(), rtol=0.2, err_msg=name
            )


def _initial_states(name):
    """A fresh Keras model's weights (as a PyTorch state) and a fresh PyTorch model."""
    from dlomix.constants import alphabet
    from dlomix.models.chargestate import ChargeStatePredictor
    from dlomix.models.chargestate_torch import (
        ChargeStatePredictor as ChargeStatePredictorTorch,
    )
    from dlomix.models.deepLC import DeepLCRetentionTimePredictor
    from dlomix.models.detectability import DetectabilityModel
    from dlomix.models.detectability_torch import (
        DetectabilityModel as DetectabilityModelTorch,
    )
    from dlomix.models.ionmob import Ionmob
    from dlomix.models.prosit import PrositRetentionTimePredictor
    from dlomix.models.prosit_torch import (
        PrositRetentionTimePredictor as PrositRetentionTimePredictorTorch,
    )

    if name == "deeplc":
        inputs = _deeplc_inputs(True)
        model_tf = DeepLCRetentionTimePredictor(use_global_features=True)
        model_torch = DeepLCRetentionTimePredictorTorch(use_global_features=True)
        model_tf(inputs)
        model_torch({k: _to_torch(v) for k, v in inputs.items()})
        return _deeplc_state(model_tf, model_torch), model_torch
    if name == "ionmob":
        model_tf = Ionmob(num_tokens=NUM_TOKENS)
        model_tf(_ionmob_inputs())
        return _ionmob_state(model_tf), IonmobTorch(num_tokens=NUM_TOKENS)
    if name == "detectability":
        sequences = _padded_sequences(len(alphabet), seq_length=40)
        model_tf = DetectabilityModel(num_units=64)
        model_torch = DetectabilityModelTorch(num_units=64, alphabet_size=len(alphabet))
        model_tf(sequences)
        model_torch(_to_torch(sequences))
        return _detectability_state(model_tf), model_torch

    sequences = _padded_sequences(len(ALPHABET))
    model_cls_tf, model_cls_torch = {
        "prosit_rt": (PrositRetentionTimePredictor, PrositRetentionTimePredictorTorch),
        "chargestate": (ChargeStatePredictor, ChargeStatePredictorTorch),
    }[name]
    model_tf = model_cls_tf(alphabet=ALPHABET)
    model_torch = model_cls_torch(alphabet=ALPHABET)
    model_tf(sequences)
    model_torch(_to_torch(sequences))
    return {**_encoder_state(model_tf), **_regressor_state(model_tf)}, model_torch


@requires_tensorflow_keras_backend
@pytest.mark.parametrize(
    "name", ["prosit_rt", "chargestate", "detectability", "deeplc", "ionmob"]
)
def test_torch_models_are_initialized_like_keras(name):
    _assert_same_initial_distribution(*_initial_states(name))


def test_torch_prosit_intensity_is_initialized_like_keras():
    # Prosit intensity has no weight mapping here (see scripts/check_backend_parity.py),
    # so check the initializers directly
    from dlomix.layers.keras_initializers_torch import EMBEDDING_INIT_RANGE
    from dlomix.models.prosit_torch import PrositIntensityPredictor

    model = PrositIntensityPredictor(seq_length=30)  # 32 tokens with the termini
    model(torch.randint(low=1, high=20, size=(2, 32)))
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            assert module.weight.abs().max() <= EMBEDDING_INIT_RANGE, name
        elif isinstance(module, (torch.nn.Linear, torch.nn.GRU)):
            for parameter_name, parameter in module.named_parameters():
                if parameter_name.startswith("bias"):
                    assert not torch.any(parameter), f"{name}.{parameter_name}"
                elif parameter_name.startswith("weight_hh"):
                    # Keras' orthogonal recurrent kernel
                    gram = parameter.detach().T @ parameter.detach()
                    assert torch.allclose(
                        gram, torch.eye(gram.shape[0]), atol=1e-5
                    ), f"{name}.{parameter_name}"


@requires_tensorflow_keras_backend
def test_tf_models_use_standard_gru_kernel_on_apple_gpu(monkeypatch):
    from dlomix.layers import gru_kernel
    from dlomix.models.chargestate import ChargeStatePredictor
    from dlomix.models.detectability import DetectabilityModel
    from dlomix.models.ionmob import Ionmob
    from dlomix.models.prosit import (
        PrositIntensityPredictor,
        PrositRetentionTimePredictor,
    )

    monkeypatch.setattr(gru_kernel, "_apple_gpu_visible", lambda: True)
    gru_kernel._warn_standard_kernel.cache_clear()  # warns once per process
    with pytest.warns(UserWarning, match="standard GRU kernel"):
        models = [
            PrositRetentionTimePredictor(),
            PrositIntensityPredictor(),
            ChargeStatePredictor(),
            DetectabilityModel(num_units=16),
            Ionmob(num_tokens=NUM_TOKENS),
        ]
    for model in models:
        grus = [
            layer
            for layer in model._flatten_layers()
            if isinstance(layer, keras.layers.GRU)
        ]
        assert grus, type(model).__name__
        assert all(gru.use_cudnn is False for gru in grus), type(model).__name__

    monkeypatch.setattr(gru_kernel, "_apple_gpu_visible", lambda: False)
    assert gru_kernel.gru_kernel_kwargs() == {}
