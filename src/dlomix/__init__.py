from .config import _BACKEND, PYTORCH_BACKEND, TENSORFLOW_BACKEND

if _BACKEND in PYTORCH_BACKEND:
    try:
        import torch  # Check for PyTorch
    except ImportError:
        raise ImportError(
            "PyTorch is selected as backend, but it is not installed. Please install DLOmix with PyTorch"
            "using  'pip install dlomix[torch]'"
            "or separately install the required backend."
        )
if _BACKEND in TENSORFLOW_BACKEND:
    try:
        import tensorflow  # Check for TensorFlow
    except ImportError:
        raise ImportError(
            "TensorFlow is the default backend or was selected, but it is not installed. Please install DLOmix with TensorFlow"
            "using  'pip install dlomix[tensorflow]'"
            "or separately install the required backend."
        )

__all__ = [
    "_BACKEND",  # Also expose the current backend name
]
