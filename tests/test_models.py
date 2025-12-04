import logging

import pytest
import tensorflow as tf

from dlomix.models.chargestate import ChargeStatePredictor
from dlomix.models.prosit import PrositIntensityPredictor, PrositRetentionTimePredictor

logger = logging.getLogger(__name__)


def test_prosit_retention_time_model():
    model = PrositRetentionTimePredictor()
    logger.info(model)
    assert model is not None


def test_prosit_intensity_model():
    model = PrositIntensityPredictor(
        input_keys={
            "SEQUENCE_KEY": "sequence",
        },
        meta_data_keys=["collision_energy", "precursor_charge"],
    )

    seq_len = model.seq_length
    dummy_input = {
        "sequence": tf.zeros((1, seq_len), dtype=tf.int32),
        "collision_energy": tf.zeros((1, 1), dtype=tf.float32),
        "precursor_charge": tf.zeros((1, 6), dtype=tf.float32),
    }
    _ = model(dummy_input)
    model.summary(print_fn=logger.info)
    logger.info(model)
    assert model is not None


def test_prosit_intensity_model_ptm_on_input():
    model = PrositIntensityPredictor(
        input_keys={
            "SEQUENCE_KEY": "sequence",
        },
        meta_data_keys=["collision_energy", "precursor_charge", "fragmentation_type"],
        use_prosit_ptm_features=True,
    )

    seq_len = model.seq_length

    dummy_input = {
        "sequence": tf.zeros((1, seq_len), dtype=tf.int32),
        "collision_energy": tf.zeros((1, 1), dtype=tf.float32),
        "precursor_charge": tf.zeros((1, 6), dtype=tf.float32),
        "fragmentation_type": tf.zeros((1, 1), dtype=tf.float32),
        PrositIntensityPredictor.PTM_INPUT_KEYS[0]: tf.zeros(
            (1, seq_len, 6), dtype=tf.float32
        ),
        PrositIntensityPredictor.PTM_INPUT_KEYS[1]: tf.zeros(
            (1, seq_len, 6), dtype=tf.float32
        ),
        PrositIntensityPredictor.PTM_INPUT_KEYS[2]: tf.zeros(
            (1, seq_len, 1), dtype=tf.float32
        ),
    }
    _ = model(dummy_input)

    model.summary(print_fn=logger.info)
    logger.info(model.input_keys)
    logger.info(model)
    assert model is not None


def test_prosit_intensity_model_ptm_on_missing():
    model = PrositIntensityPredictor(use_prosit_ptm_features=True)
    seq_len = model.seq_length
    with pytest.warns(UserWarning, match="PTM"):
        dummy_input = {
            "sequence": tf.zeros((1, seq_len), dtype=tf.int32),
            "collision_energy": tf.zeros((1, 1), dtype=tf.float32),
            "precursor_charge": tf.zeros((1, 6), dtype=tf.float32),
            "fragmentation_type": tf.zeros((1, 1), dtype=tf.float32),
        }

        _ = model(dummy_input)


def test_prosit_intensity_model_encoding_metadata_missing():
    model = PrositIntensityPredictor()
    seq_len = model.seq_length

    with pytest.raises(ValueError):
        dummy_input = {
            "sequence": tf.zeros((1, seq_len), dtype=tf.int32),
        }
        _ = model(dummy_input)


def basic_model_existence_test(model):
    logger.info(model)
    assert model is not None

    # Explicitly build the model with a dummy input shape (1, seq_length)
    dummy_input = tf.zeros((1, 30), dtype=tf.int32)
    _ = model(dummy_input)
    assert len(model.trainable_weights) > 0


def test_dominant_chargestate_model():
    model = ChargeStatePredictor(model_flavour="dominant")
    basic_model_existence_test(model)


def test_observed_chargestate_model():
    model = ChargeStatePredictor(model_flavour="observed")
    basic_model_existence_test(model)


def test_chargestate_distribution_model():
    model = ChargeStatePredictor(model_flavour="relative")
    basic_model_existence_test(model)
