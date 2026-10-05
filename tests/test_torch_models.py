import logging

import keras
import pytest
import torch

from dlomix.models.chargestate import ChargeStatePredictor
from dlomix.models.chargestate_torch import (
    ChargeStatePredictor as ChargeStatePredictorTorch,
)
from dlomix.models.prosit import PrositIntensityPredictor, PrositRetentionTimePredictor
from dlomix.models.prosit_torch import (
    PrositIntensityPredictor as PrositIntensityPredictorTorch,
)
from dlomix.models.prosit_torch import (
    PrositRetentionTimePredictor as PrositRetentionTimePredictorTorch,
)

logger = logging.getLogger(__name__)

# In TF 2.16+ `tf.keras` *is* Keras 3, whose backend is a single global setting.
# So the "TensorFlow" models only build TensorFlow layers while Keras is running
# on the TensorFlow backend. The tests that construct a TF model and a Torch model
# side by side therefore need KERAS_BACKEND=tensorflow, which is what
# DLOMIX_BACKEND=tensorflow (the default) selects. The torch-only tests below run
# under either setting.
requires_tensorflow_keras_backend = pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason=(
        "compares a tf.keras model against a torch model in one process, which "
        f"needs the TensorFlow Keras backend (currently '{keras.backend.backend()}')"
    ),
)


def basic_model_existence_test_torch(model):
    logger.info(model)
    assert model is not None

    assert len(list(model.parameters())) > 0


# ------------------ CS | check for existence of model & its parameters ------------------


def test_dominant_chargestate_model_torch():
    model = ChargeStatePredictorTorch(model_flavour="dominant")
    basic_model_existence_test_torch(model)


def test_observed_chargestate_model_torch():
    model = ChargeStatePredictorTorch(model_flavour="observed")
    basic_model_existence_test_torch(model)


def test_chargestate_distribution_model_torch():
    model = ChargeStatePredictorTorch(model_flavour="relative")
    basic_model_existence_test_torch(model)


def test_chargestate_unknown_flavour_raises_torch():
    with pytest.raises(ValueError, match="model_flavour"):
        ChargeStatePredictorTorch(model_flavour="dominnant")


# ------------------ CS | comparison of tf & torch ------------------


@requires_tensorflow_keras_backend
def test_tf_torch_equivalence_chargestate_model_shapes():
    # to compare tf & torch: input & output shapes at beginnin & end of 1 forward

    batch_size = 2
    seq_len = 30

    dummy_input_torch = torch.randint(low=0, high=15, size=(batch_size, seq_len))
    dummi_input_tf = dummy_input_torch.numpy()

    model_tf = ChargeStatePredictor(model_flavour="dominant", seq_length=seq_len)
    model_torch = ChargeStatePredictorTorch(
        model_flavour="dominant", seq_length=seq_len
    )

    output_tf = model_tf(dummi_input_tf)
    output_torch = model_torch(dummy_input_torch)

    assert output_tf.shape == output_torch.detach().numpy().shape


# -------------- Prosit RT | check for existence of model & its parameters -------


def test_RT_model_torch():
    model = PrositRetentionTimePredictorTorch()
    basic_model_existence_test_torch(model)


@pytest.mark.parametrize(
    "model_cls", [PrositRetentionTimePredictorTorch, ChargeStatePredictorTorch]
)
def test_attention_length_taken_from_first_input_torch(model_cls):
    # seq_length does not have to match the padded width, as in TensorFlow
    model = model_cls(seq_length=30).eval()
    sequences = torch.randint(low=1, high=15, size=(2, 32))
    output = model(sequences)
    assert model.attention.seq_len == 32

    # a fresh model loads the weights without a forward pass first
    reloaded = model_cls(seq_length=30).eval()
    reloaded.load_state_dict(model.state_dict())
    assert torch.equal(reloaded(sequences), output)

    with pytest.raises(ValueError, match="length 32, got length 34"):
        model(torch.randint(low=1, high=15, size=(2, 34)))


# -------------- Prosit RT | comparison of tf & torch ----------------------


@requires_tensorflow_keras_backend
def test_tf_torch_equivalence_RT_model_shapes():
    # to compare tf & torch: input & output shapes at beginnin & end of 1 forward

    batch_size = 2
    seq_len = 30

    dummy_input_torch = torch.randint(low=0, high=15, size=(batch_size, seq_len))
    dummi_input_tf = dummy_input_torch.numpy()

    model_tf = PrositRetentionTimePredictor(seq_length=seq_len)
    model_torch = PrositRetentionTimePredictorTorch(seq_length=seq_len)

    output_tf = model_tf(dummi_input_tf)
    output_torch = model_torch(dummy_input_torch)

    assert output_tf.shape == output_torch.detach().numpy().shape


# -------------- Prosit Intensity | check for existence of model & its parameters -------
def test_intensity_model_torch():
    model = PrositIntensityPredictorTorch()
    basic_model_existence_test_torch(model)


# -------------- Prosit Intensity | comparison of tf & torch ----------------------
@requires_tensorflow_keras_backend
def test_tf_torch_equivalence_intensity_model_shapes():
    # to compare tf & torch: input & output shapes at beginnin & end of 1 forward

    batch_size = 2
    seq_len = 30

    dummy_input_torch = torch.randint(low=0, high=15, size=(batch_size, seq_len))
    dummi_input_tf = dummy_input_torch.numpy()

    model_tf = PrositIntensityPredictor(
        seq_length=seq_len,
        with_termini=False,
    )

    model_torch = PrositIntensityPredictorTorch(
        seq_length=seq_len,
        with_termini=False,
    )

    output_tf = model_tf(dummi_input_tf)
    output_torch = model_torch(dummy_input_torch)

    assert output_tf.shape == output_torch.detach().numpy().shape
