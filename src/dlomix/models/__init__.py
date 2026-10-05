from ..config import _BACKEND, PYTORCH_BACKEND, TENSORFLOW_BACKEND

__all__ = []

if _BACKEND in TENSORFLOW_BACKEND:
    from .base import RetentionTimePredictor
    from .chargestate import ChargeStatePredictor
    from .deepLC import DeepLCRetentionTimePredictor
    from .detectability import DetectabilityModel
    from .ionmob import Ionmob
    from .model_utils import (
        download_remote_model_weights,
        load_and_adapt_pretrained_model,
    )
    from .prosit import PrositIntensityPredictor, PrositRetentionTimePredictor

    # TensorFlow models only
    __all__.append("RetentionTimePredictor")

    # TensorFlow utility functions
    __all__.append("load_and_adapt_pretrained_model")
    __all__.append("download_remote_model_weights")


elif _BACKEND in PYTORCH_BACKEND:
    from .chargestate_torch import ChargeStatePredictor
    from .deepLC_torch import DeepLCRetentionTimePredictor
    from .detectability_torch import DetectabilityModel
    from .ionmob_torch import Ionmob
    from .prosit_torch import PrositIntensityPredictor, PrositRetentionTimePredictor

__all__.extend(
    [
        "ChargeStatePredictor",
        "PrositRetentionTimePredictor",
        "PrositIntensityPredictor",
        "DetectabilityModel",
        "DeepLCRetentionTimePredictor",
        "Ionmob",
    ]
)
