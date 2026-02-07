from ..config import _BACKEND, PYTORCH_BACKEND, TENSORFLOW_BACKEND

__all__ = []

if _BACKEND in TENSORFLOW_BACKEND:
    from .base import RetentionTimePredictor
    from .chargestate import ChargeStatePredictor
    from .detectability import DetectabilityModel
    from .model_utils import load_and_adapt_pretrained_model
    from .prosit import PrositIntensityPredictor, PrositRetentionTimePredictor

    # TensorFlow models only
    __all__.append("RetentionTimePredictor")

    # TensorFlow utility functions
    __all__.append("load_and_adapt_pretrained_model")


elif _BACKEND in PYTORCH_BACKEND:
    from .chargestate_torch import ChargeStatePredictor
    from .detectability_torch import DetectabilityModel
    from .ionmob_torch import Ionmob
    from .prosit_torch import PrositIntensityPredictor, PrositRetentionTimePredictor

    # PyTorch models only
    __all__.append("Ionmob")

__all__.extend(
    [
        "ChargeStatePredictor",
        "PrositRetentionTimePredictor",
        "PrositIntensityPredictor",
        "DetectabilityModel",
        "ChargeStatePredictor",
    ]
)
