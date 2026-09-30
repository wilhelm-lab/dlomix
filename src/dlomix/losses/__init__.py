from ..config import _BACKEND, PYTORCH_BACKEND

# The intensity losses are backend-agnostic (keras.ops) and shared by both backends.
from .intensity import masked_pearson_correlation_distance, masked_spectral_distance

__all__ = ["masked_pearson_correlation_distance", "masked_spectral_distance"]

if _BACKEND in PYTORCH_BACKEND:
    # Ionmob is a PyTorch-only model, and its loss is a stateful nn.Module.
    from .ionmob_torch import MaskedIonmobLoss

    __all__.append("MaskedIonmobLoss")
