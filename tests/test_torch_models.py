import logging

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


# ------------------ CS | comparison of tf & torch ------------------


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


# -------------- Prosit RT | comparison of tf & torch ----------------------


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
def test_tf_torch_equivalence_intensity_model_shapes():
    # to compare tf & torch: input & output shapes at beginnin & end of 1 forward

    batch_size = 2
    seq_len = 30

    dummy_input_torch = torch.randint(low=0, high=15, size=(batch_size, seq_len))
    dummi_input_tf = dummy_input_torch.numpy()

    model_tf = PrositIntensityPredictor()
    model_torch = PrositIntensityPredictorTorch()

    output_tf = model_tf(dummi_input_tf)
    output_torch = model_torch(dummy_input_torch)

    assert output_tf.shape == output_torch.detach().numpy().shape
