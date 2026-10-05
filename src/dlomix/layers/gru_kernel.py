"""Choose the GRU kernel for the TensorFlow models.

On Apple GPUs (``tensorflow-metal``), Keras runs ``GRU`` layers through a fused Metal
kernel by default (``use_cudnn="auto"``). That kernel computes a different function
when the GRU biases are not zero, so weights trained on a Mac give other predictions
on CPU, on Linux/CUDA or in PyTorch, and published weights give wrong predictions on
a Mac (seen with tensorflow-metal 1.2.0). DLOmix therefore uses the standard GRU
kernel when an Apple GPU is visible, which is correct everywhere but slow on the GPU:
training Prosit intensity on an M1 Max took 622 s per epoch, against 78 s on the CPU
(and 63 s with the fused kernel). For GRU models on a Mac, train on the CPU::

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # before TensorFlow uses the GPU

or use the PyTorch backend, which trains on the Apple GPU (16 s per epoch).

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
    # on macOS, TensorFlow only sees a GPU through the tensorflow-metal plugin; a GPU
    # hidden with tf.config.set_visible_devices does not count
    return platform.system() == "Darwin" and bool(tf.config.get_visible_devices("GPU"))


@functools.lru_cache(maxsize=None)
def _warn_standard_kernel() -> None:
    warnings.warn(
        "An Apple GPU (tensorflow-metal) is visible: DLOmix uses the standard GRU "
        "kernel, because the fused Metal kernel computes a different function once "
        "the GRU biases are not zero. The standard kernel trains about 8x slower on "
        "the Apple GPU than on the CPU: call "
        'tf.config.set_visible_devices([], "GPU") before TensorFlow uses the GPU, '
        "or use the PyTorch backend. See dlomix.layers.gru_kernel.",
        stacklevel=3,
    )


def gru_kernel_kwargs() -> dict:
    """Keyword arguments for every ``GRU`` layer of the DLOmix TensorFlow models."""
    if not _apple_gpu_visible():
        return {}
    _warn_standard_kernel()
    return {"use_cudnn": False}
