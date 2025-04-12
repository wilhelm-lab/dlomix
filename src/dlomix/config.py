import os
import warnings

DEFAULT_BACKEND = "tensorflow"
TENSORFLOW_BACKEND = ["tensorflow", "tf"]
PYTORCH_BACKEND = ["pytorch", "torch", "pt"]


# Allow setting backend via environment variable or default to tensorflow
_BACKEND = os.environ.get("DLOMIX_BACKEND", DEFAULT_BACKEND).lower()

if _BACKEND in TENSORFLOW_BACKEND:
    BACKEND_PRETTY_NAME = "TensorFlow"
elif _BACKEND in PYTORCH_BACKEND:
    BACKEND_PRETTY_NAME = "PyTorch"
else:
    raise ValueError(
        f"DLOmix supports 'tensorflow' or 'pytorch' backends, got '{_BACKEND}'"
    )

message = f"""

Using {BACKEND_PRETTY_NAME} Backend for DLOmix.
To change the backend, set the DLOMIX_BACKEND environment variable to tensorflow or pytorch and re-import DLOmix."

"""


def custom_show_warning(message, category, filename, lineno, file=None, line=None):
    print(f"{message}")


with warnings.catch_warnings():
    warnings.simplefilter("once", category=UserWarning)  # Show only once
    warnings.showwarning = (
        custom_show_warning  # Temporarily override the default showwarning
    )
    warnings.warn(message, UserWarning)  # Trigger the warning
