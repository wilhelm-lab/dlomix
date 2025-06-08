import os
import warnings

DEFAULT_BACKEND = "tensorflow"
BACKEND_PRETTY_NAME = "TensorFlow"

TENSORFLOW_BACKEND = ["tensorflow", "tf"]
PYTORCH_BACKEND = ["pytorch", "torch", "pt"]


def custom_show_warning(msg, category, filename, lineno, file=None, line=None):
    print(f"{msg}")


# Allow setting backend via environment variable or default to tensorflow
_BACKEND = os.environ.get("DLOMIX_BACKEND", DEFAULT_BACKEND).lower().strip()

if _BACKEND in TENSORFLOW_BACKEND:
    BACKEND_PRETTY_NAME = "TensorFlow"
elif _BACKEND in PYTORCH_BACKEND:
    BACKEND_PRETTY_NAME = "PyTorch"
else:
    # default to TensorFlow
    with warnings.catch_warnings():
        warnings.simplefilter("once", category=UserWarning)  # Show only once
        warnings.showwarning = (
            custom_show_warning  # Temporarily override the default showwarning
        )
        warnings.warn(
            f"Backend '{_BACKEND}' is not supported. Defaulting to {BACKEND_PRETTY_NAME} backend.",
            UserWarning,
        )

    _BACKEND = DEFAULT_BACKEND


with warnings.catch_warnings():
    warnings.simplefilter("once", category=UserWarning)  # Show only once
    warnings.showwarning = (
        custom_show_warning  # Temporarily override the default showwarning
    )

    message = f"Using {BACKEND_PRETTY_NAME} Backend for DLOmix. To change the backend, set the DLOMIX_BACKEND environment variable to tensorflow or pytorch and re-import DLOmix."
    warnings.warn(message, UserWarning)  # Trigger the warning
