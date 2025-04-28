from ..config import _BACKEND, PYTORCH_BACKEND, TENSORFLOW_BACKEND

__all__ = []

if _BACKEND in TENSORFLOW_BACKEND:
    from .intensity import masked_pearson_correlation_distance, masked_spectral_distance

elif _BACKEND in PYTORCH_BACKEND:
    from .intensity_torch import (
        masked_pearson_correlation_distance,
        masked_spectral_distance,
    )
    from .ionmob_torch import MaskedIonmobLoss

    __all__.append(
        "MaskedIonmobLoss",
    )

__all__.extend(["masked_pearson_correlation_distance", "masked_spectral_distance"])
