import logging

import pytest

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
    model.build(
        {
            "sequence": (
                None,
                seq_len,
            ),
            "collision_energy": (None, 1),
            "precursor_charge": (None, 6),
        }
    )
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
    model.build(
        {
            "sequence": (
                None,
                seq_len,
            ),
            "collision_energy": (None, 1),
            "precursor_charge": (None, 6),
            "fragmentation_type": (None, 1),
            PrositIntensityPredictor.PTM_INPUT_KEYS[0]: (
                None,
                seq_len,
                6,
            ),
            PrositIntensityPredictor.PTM_INPUT_KEYS[1]: (
                None,
                seq_len,
                6,
            ),
            PrositIntensityPredictor.PTM_INPUT_KEYS[2]: (
                None,
                seq_len,
                1,
            ),
        }
    )
    model.summary(print_fn=logger.info)
    logger.info(model.input_keys)
    logger.info(model)
    assert model is not None


def test_prosit_intensity_model_ptm_on_missing():
    model = PrositIntensityPredictor(use_prosit_ptm_features=True)
    seq_len = model.seq_length
    with pytest.raises(ValueError, match="PTM"):
        model.build(
            {
                "sequence": (
                    None,
                    seq_len,
                ),
                "collision_energy": (None,),
                "precursor_charge": (None, 6),
                "fragmentation_type": (None,),
                # no PTM features
            }
        )


def test_prosit_intensity_model_encoding_metadata_missing():
    model = PrositIntensityPredictor(
        meta_data_keys=["meta_data_1", "meta_data_2"], use_meta_data=True
    )
    seq_len = model.seq_length

    with pytest.raises(ValueError, match="metadata"):
        model.build(
            {
                "sequence": (
                    None,
                    seq_len,
                ),
                # no meta-data while expected
            }
        )


def test_prosit_intensity_model_no_metadata():
    model = PrositIntensityPredictor(
        input_keys={
            "SEQUENCE_KEY": "sequence",
        },
        meta_data_keys=None,
    )

    seq_len = model.seq_length

    assert model.meta_data_keys == []

    model.build(
        {
            "sequence": (
                None,
                seq_len,
            ),
        }
    )

    assert model is not None
    assert model.meta_encoder is None


def basic_model_existence_test(model):
    logger.info(model)
    assert model is not None

    # Explicitly build the model with a dummy input shape (batch_size, seq_length)
    model.build((None, 30))
    assert len(model.trainable_variables) > 0


def test_dominant_chargestate_model():
    model = ChargeStatePredictor(model_flavour="dominant")
    basic_model_existence_test(model)


def test_observed_chargestate_model():
    model = ChargeStatePredictor(model_flavour="observed")
    basic_model_existence_test(model)


def test_chargestate_distribution_model():
    model = ChargeStatePredictor(model_flavour="relative")
    basic_model_existence_test(model)
