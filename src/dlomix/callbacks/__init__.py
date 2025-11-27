from ..config import _BACKEND, PYTORCH_BACKEND, TENSORFLOW_BACKEND

__all__ = []

if _BACKEND in TENSORFLOW_BACKEND:
    from .cyclic_lr import CyclicLR

    __all__.append("CyclicLR")

elif _BACKEND in PYTORCH_BACKEND:
    pass
