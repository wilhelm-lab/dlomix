
from dlomix.models import PrositRetentionTimePredictor, PrositIntensityPredictor
import logging

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
            "sequence": (None, seq_len,),
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
            "sequence": (None, seq_len,),
            "collision_energy": (None, 1),
            "precursor_charge": (None, 6),
            "fragmentation_type": (None, 1),
            "ptm_atom_count_loss": (None, seq_len, 6,),
            "ptm_atom_count_gain": (None, seq_len, 6,),
        }
    )
    model.summary(print_fn=logger.info)
    logger.info(model.input_keys)
    logger.info(model)
    assert model is not None



