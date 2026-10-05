from ..config import _BACKEND, PYTORCH_BACKEND

# The intensity losses are backend-agnostic (keras.ops) and shared by both backends.
from .intensity import masked_pearson_correlation_distance, masked_spectral_distance

__all__ = ["masked_pearson_correlation_distance", "masked_spectral_distance"]

# MaskedIonmobLoss has the same name and signature on both backends: a keras.ops
# implementation for TensorFlow, and the stateful nn.Module for PyTorch.
if _BACKEND in PYTORCH_BACKEND:
    from .ionmob_torch import MaskedIonmobLoss
else:
    from .ionmob import MaskedIonmobLoss

__all__.append("MaskedIonmobLoss")
