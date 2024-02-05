
from dlomix.models import PrositRetentionTimePredictor, PrositIntensityPredictor
import logging

logger = logging.getLogger(__name__)



def test_prosit_retention_time_model():
    model = PrositRetentionTimePredictor()
    logger.info(model)
    assert model is not None

def test_prosit_intensity_model():
    model = PrositIntensityPredictor()
    logger.info(model)
    assert model is not None



