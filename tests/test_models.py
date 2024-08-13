import logging

import pytest

from dlomix.models import PrositIntensityPredictor, PrositRetentionTimePredictor

logger = logging.getLogger(__name__)


def test_prosit_retention_time_model():
    model = PrositRetentionTimePredictor()
    logger.info(model)
    assert model is not None


def test_prosit_intensity_model():
    model = PrositIntensityPredictor()
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
    assert model.ptm_encoder is None
    assert model.ptm_aa_fusion is None


def test_prosit_intensity_model_ptm_on():
    model = PrositIntensityPredictor(use_ptm_counts=True)
    logger.info(model.input_keys)
    logger.info(model)
    assert model is not None
    assert model.ptm_encoder is not None
    assert model.ptm_aa_fusion is not None


def test_prosit_intensity_model_ptm_on_input():
    model = PrositIntensityPredictor(use_ptm_counts=True)
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
    model = PrositIntensityPredictor(use_ptm_counts=True)
    seq_len = model.seq_length
    with pytest.raises(ValueError):
        model.build(
            {
                "sequence": (
                    None,
                    seq_len,
                ),
                "collision_energy": (None, 1),
                "precursor_charge": (None, 6),
                "fragmentation_type": (None, 1),
                # no PTM features
            }
        )


def test_prosit_intensity_model_encoding_metadata_missing():
    model = PrositIntensityPredictor()
    seq_len = model.seq_length

    with pytest.raises(ValueError):
        model.build(
            {
                "sequence": (
                    None,
                    seq_len,
                ),
                # no meta-data while expected
            }
        )
