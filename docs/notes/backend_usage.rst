Backend Usage Guide
*******************

DLOmix provides a unified API that works with both TensorFlow and PyTorch backends. This guide explains how the import system works and how to use the correct classes in your code.

Importing Classes
*******************

When importing classes from DLOmix, you should always use the top-level module:

.. code-block:: python

   # Correct way to import
   from dlomix.models import ChargeStatePredictor
   from dlomix.layers import AttentionLayer

   # The backend implementation (TensorFlow or PyTorch) will be selected automatically
   # based on your configuration

The class names in your code will always be the same (e.g., `ChargeStatePredictor`, `AttentionLayer`), regardless of which backend is being used. DLOmix handles the backend selection automatically.

If you are interested in the implementation details, you can find them under the respective modules with the following naming convention:
- src/dlomix/MODULE_NAME/SUBMODULE.py (for TensorFlow, always with no suffix)
- src/dlomix/MODULE_NAME/SUBMODULE_torch.py (for PyTorch, always with a `_torch` suffix)

Note that some modules are shared across both backends, hence they do not have any files with `_torch` suffix.


How Backend Selection Works
****************************

1. **Automatic Detection**: DLOmix automatically detects which backends are available in your environment.

2. **Default Selection**: If both TensorFlow and PyTorch are installed, TensorFlow will be used by default.

3. **Environment Variables**: You can set the backend via environment variables:

   .. code-block:: bash

      export DLOMIX_BACKEND=tensorflow # default behavior if env variable is not set
      export DLOMIX_BACKEND=pytorch


   This must be set before importing DLOmix or any of its modules in the user code.
