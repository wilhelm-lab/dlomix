import os
import sys
import warnings

DEFAULT_BACKEND = "tensorflow"
BACKEND_PRETTY_NAME = "TensorFlow"

TENSORFLOW_BACKEND = ["tensorflow", "tf"]
PYTORCH_BACKEND = ["pytorch", "torch", "pt"]


def custom_show_warning(msg, category, filename, lineno, file=None, line=None):
    print(f"{msg}")


def _warn(message):
    """Emit a one-off UserWarning as a plain message, without the traceback noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("once", category=UserWarning)
        warnings.showwarning = custom_show_warning
        warnings.warn(message, UserWarning)


def _align_keras_backend(backend):
    """Point Keras at the same framework DLOmix is using.

    DLOmix implements its losses and metrics once against ``keras.ops`` (see
    :mod:`dlomix.losses` / :mod:`dlomix.eval`), so Keras has to run on the backend
    the user selected. Keras resolves ``KERAS_BACKEND`` once, on first import, so
    this must happen before anything imports keras -- and an explicit
    ``KERAS_BACKEND`` from the user takes precedence.
    """
    keras_backend = "torch" if backend in PYTORCH_BACKEND else "tensorflow"
    os.environ.setdefault("KERAS_BACKEND", keras_backend)

    # Too late to set the variable if keras is already loaded. Say so now; otherwise
    # this only surfaces much later as an opaque tensor-type error.
    already_imported = sys.modules.get("keras")
    if already_imported and already_imported.backend.backend() != keras_backend:
        _warn(
            f"keras was already imported with the '{already_imported.backend.backend()}' "
            f"backend before DLOmix, which expects '{keras_backend}' for "
            f"DLOMIX_BACKEND='{backend}'. The backend cannot be changed after import. "
            "Import dlomix before keras, or set KERAS_BACKEND yourself."
        )


# Allow setting backend via environment variable or default to tensorflow
_BACKEND = os.environ.get("DLOMIX_BACKEND", DEFAULT_BACKEND).lower().strip()

if _BACKEND in TENSORFLOW_BACKEND:
    BACKEND_PRETTY_NAME = "TensorFlow"
elif _BACKEND in PYTORCH_BACKEND:
    BACKEND_PRETTY_NAME = "PyTorch"
else:
    _warn(
        f"Backend '{_BACKEND}' is not supported. Defaulting to {BACKEND_PRETTY_NAME} backend."
    )
    _BACKEND = DEFAULT_BACKEND

_align_keras_backend(_BACKEND)

_warn(
    f"Using {BACKEND_PRETTY_NAME} Backend for DLOmix. To change the backend, set the "
    "DLOMIX_BACKEND environment variable to tensorflow or pytorch and re-import DLOmix."
)
