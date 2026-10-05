"""Choose the GRU kernel for the TensorFlow models.

On Apple GPUs (``tensorflow-metal``), Keras runs ``GRU`` layers through a fused Metal
kernel by default (``use_cudnn="auto"``). That kernel computes a different function
when the GRU biases are not zero, so weights trained on a Mac give other predictions
on CPU, on Linux/CUDA or in PyTorch, and published weights give wrong predictions on
a Mac (seen with tensorflow-metal 1.2.0). DLOmix therefore uses the standard GRU
kernel when an Apple GPU is visible, which is correct everywhere but slower.

To use the fused Metal kernel anyway, set ``use_cudnn = "auto"`` on the model's GRU
layers after building it::

    for layer in model._flatten_layers():
        if isinstance(layer, keras.layers.GRU):
            layer.use_cudnn = "auto"
"""

import functools
import platform
import warnings

import tensorflow as tf


@functools.lru_cache(maxsize=None)
def _apple_gpu_visible() -> bool:
    # on macOS, TensorFlow only sees a GPU through the tensorflow-metal plugin
    return platform.system() == "Darwin" and bool(
        tf.config.list_physical_devices("GPU")
    )


@functools.lru_cache(maxsize=None)
def _warn_standard_kernel() -> None:
    warnings.warn(
        "An Apple GPU (tensorflow-metal) is visible: DLOmix uses the standard GRU "
        "kernel instead of the fused Metal kernel, which computes a different "
        "function when the GRU biases are not zero. See dlomix.layers.gru_kernel.",
        stacklevel=3,
    )


def gru_kernel_kwargs() -> dict:
    """Keyword arguments for every ``GRU`` layer of the DLOmix TensorFlow models."""
    if not _apple_gpu_visible():
        return {}
    _warn_standard_kernel()
    return {"use_cudnn": False}
