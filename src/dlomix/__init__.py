from ._metadata import __version__
from .config import _BACKEND, BACKEND_PRETTY_NAME, PYTORCH_BACKEND, TENSORFLOW_BACKEND

BACKEND_IMPORT_ERROR_MESSAGE = f"""
{BACKEND_PRETTY_NAME} is selected as backend, but it is not installed.
Please install DLOmix with {BACKEND_PRETTY_NAME} using 'pip install dlomix[{BACKEND_PRETTY_NAME.lower()}]'
or separately install the required backend via pip.
"""


if _BACKEND in PYTORCH_BACKEND:
    try:
        import torch  # Check for PyTorch
    except ImportError:
        raise ImportError(BACKEND_IMPORT_ERROR_MESSAGE)
if _BACKEND in TENSORFLOW_BACKEND:
    try:
        import tensorflow  # Check for TensorFlow
    except ImportError:
        raise ImportError(BACKEND_IMPORT_ERROR_MESSAGE)

__all__ = [
    "_BACKEND",  # Also expose the current backend name
    "__version__",  # Expose the version of the package
]
