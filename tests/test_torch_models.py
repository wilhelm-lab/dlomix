"""The PyTorch models build and behave as documented.

They compute the same function as the TensorFlow models; that is checked with
copied weights in ``test_backend_equivalence.py``.
"""

import logging

import pytest
import torch

from dlomix.models.chargestate_torch import (
    ChargeStatePredictor as ChargeStatePredictorTorch,
)
from dlomix.models.prosit_torch import (
    PrositIntensityPredictor as PrositIntensityPredictorTorch,
)
from dlomix.models.prosit_torch import (
    PrositRetentionTimePredictor as PrositRetentionTimePredictorTorch,
)

logger = logging.getLogger(__name__)


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


# -------------- Prosit Intensity | check for existence of model & its parameters -------
def test_intensity_model_torch():
    model = PrositIntensityPredictorTorch()
    basic_model_existence_test_torch(model)


def test_prosit_intensity_torch_warns_when_metadata_keys_are_ignored():
    # without use_meta_data=True the model has no metadata encoder and silently
    # ignored these inputs
    with pytest.warns(UserWarning, match="use_meta_data=False"):
        PrositIntensityPredictorTorch(meta_data_keys=["collision_energy"])


def test_intensity_model_torch_from_dataset():
    from types import SimpleNamespace

    alphabet = {"-": 0, "X": 1, "A": 2, "C": 3}
    dataset = SimpleNamespace(  # the attributes from_dataset reads
        sequence_column="peptide",
        extended_alphabet=alphabet,
        max_seq_len=10,
        with_termini=False,
        model_features=["collision_energy"],
    )

    model = PrositIntensityPredictorTorch.from_dataset(
        dataset, recurrent_layers_sizes=(8, 16)
    )

    assert model.alphabet == alphabet
    assert (model.raw_seq_length, model.with_termini) == (10, False)
    assert model.input_keys == {"SEQUENCE_KEY": "peptide"}
    assert (model.use_meta_data, model.meta_data_keys) == (True, ["collision_energy"])
    assert model.recurrent_layers_sizes == (8, 16)
