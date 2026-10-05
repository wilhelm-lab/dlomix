Installation
************

.. caution::

  DLOmix is under active development (beta). The API may still change between minor versions; see the `changelog <https://github.com/wilhelm-lab/dlomix/blob/main/CHANGELOG.md>`_ when upgrading. If you have feedback, ideas for improvements, or if you find a bug, please open an issue on GitHub.

Requirements
************

- Python 3.11 to 3.13.
- TensorFlow backend: TensorFlow 2.18 or newer, with Keras 3 (on TensorFlow 2.16+, ``tf.keras`` is Keras 3).
- PyTorch backend: PyTorch 2.3 or newer.

For TensorFlow < 2.16 (Keras 2), use ``dlomix<0.3``.

DLOmix can be installed via pip, this installs the package and the main dependencies only, excluding the backend framework to be used (TensorFlow/Keras or PyTorch):

.. code-block:: bash

  pip install dlomix

Backend Selection
******************

DLOmix supports multiple deep learning backends. To use TensorFlow/Keras or PyTorch as a backend together with DLOmix, install with the respective command:

.. code-block:: bash

  pip install "dlomix[tensorflow]"  # shorter alternative: "dlomix[tf]"

  pip install "dlomix[pytorch]"  # shorter alternative: "dlomix[torch]"

The quotes keep shells such as zsh (the default on macOS) from interpreting the square brackets.

.. note::

   You only need to install the backend you intend to use. DLOmix does not detect installed backends: it uses the backend named by the ``DLOMIX_BACKEND`` environment variable, and defaults to TensorFlow, with a warning, when the variable is not set. Set it before importing DLOmix:

   .. code-block:: bash

      export DLOMIX_BACKEND=pytorch  # or tensorflow (the default)

.. include:: backend_usage.rst


GPU Support (Linux, NVIDIA)
***************************

**TensorFlow** needs its CUDA libraries installed explicitly. The ``tf-cuda`` extra installs them as pip packages through ``tensorflow[and-cuda]``, so no system-wide CUDA toolkit is needed, only an NVIDIA driver:

.. code-block:: bash

  pip install "dlomix[tf-cuda]"

NVIDIA publishes these CUDA wheels for Linux only. On macOS and Windows the ``tf-cuda`` extra installs the regular TensorFlow build instead of failing.

**PyTorch** needs no extra: its Linux wheels on PyPI are already CUDA builds, so ``pip install "dlomix[pytorch]"`` is enough. Recent PyTorch releases target a recent CUDA version, which requires a correspondingly recent NVIDIA driver. On an older driver, install PyTorch from the matching CUDA index given by the `PyTorch install selector <https://pytorch.org/get-started/locally/>`_, then install DLOmix.


Apple GPUs (macOS, tensorflow-metal)
************************************

With ``tensorflow-metal`` installed, TensorFlow trains on the Apple GPU. ``tensorflow-metal`` 1.2.0 only works with TensorFlow below 2.20, which DLOmix therefore installs on macOS for Python 3.11 and 3.12 (Python 3.13 has no ``tensorflow-metal`` wheel and runs TensorFlow on the CPU).

Its fused GRU kernel does not compute the standard GRU: it ignores the GRU's input bias and adds the recurrent bias outside the reset gate. Both agree while the biases are zero, as Keras initializes them, so training on a Mac looks normal. But once the biases are trained, the weights compute a different function on any other platform or in PyTorch: weights trained on a Mac give other predictions elsewhere, and published weights give wrong predictions on a Mac. This is a different formula, not floating-point error. The DLOmix TensorFlow models therefore use the standard GRU kernel when an Apple GPU is visible, and warn once. Results are then the same on every platform and in PyTorch, but the standard kernel is slow on the Apple GPU. Training Prosit intensity on an M1 Max:

.. list-table::
   :header-rows: 1

   * - Setup
     - Time per epoch
   * - TensorFlow, Apple GPU, standard GRU (default with ``tensorflow-metal``)
     - 622 s
   * - TensorFlow, CPU
     - 78 s
   * - PyTorch, Apple GPU (MPS)
     - 16 s

For the models with GRU layers (Prosit, charge state, detectability, Ionmob), train TensorFlow on the CPU by hiding the GPU before TensorFlow uses it, or use the PyTorch backend:

.. code-block:: python

  import tensorflow as tf
  tf.config.set_visible_devices([], "GPU")


Checking Your Environment
*************************

To see what your environment provides (DLOmix, Python and framework versions, the active backend, and the GPUs visible to it), run:

.. code-block:: bash

  python -m dlomix

Please include this output when reporting a bug. If a GPU is present but not detected, the usual causes are a driver that is too old for the CUDA version shown, or an ``LD_LIBRARY_PATH`` pointing at a different system CUDA installation.


Development Installation
************************

To get the develop version, you can install directly from GitHub (develop branch):

.. code-block:: bash

  pip install git+https://github.com/wilhelm-lab/dlomix.git@develop

Optional Dependencies
*********************

If you decide to use Weights & Biases for reporting, you can use the extra install command:

.. code-block:: bash

  pip install "dlomix[wandb]"

For development purposes, you can install all dependencies including both backends:

.. code-block:: bash

  pip install "dlomix[dev]"
