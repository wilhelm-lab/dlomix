Backend Usage Guide
*******************

DLOmix provides a unified API that works with both TensorFlow and PyTorch backends. This guide explains how the backend is selected, how the import system works, and how to use the correct classes in your code.

How Backend Selection Works
****************************

1. **The** ``DLOMIX_BACKEND`` **environment variable decides.** DLOmix does not detect which frameworks are installed; it reads ``DLOMIX_BACKEND`` once, when it is first imported:

   .. code-block:: bash

      export DLOMIX_BACKEND=tensorflow  # or tf; the default
      export DLOMIX_BACKEND=pytorch     # or torch, pt

2. **Default:** if the variable is not set, or set to an unknown value, DLOmix uses TensorFlow and warns.

3. **Set it before importing DLOmix.** The variable must be set before ``dlomix`` or any of its modules is imported, either in the shell or at the top of your script:

   .. code-block:: python

      import os
      os.environ["DLOMIX_BACKEND"] = "pytorch"

      import dlomix

4. **Import** ``dlomix`` **before** ``keras``. DLOmix sets the ``KERAS_BACKEND`` environment variable from ``DLOMIX_BACKEND`` (``tensorflow`` or ``torch``), unless you set it yourself. Keras fixes its backend when it is first imported, so if ``keras`` is imported before ``dlomix``, it may run on the wrong backend; DLOmix warns when that happens.

5. **One backend per process.** Neither DLOmix nor Keras can switch backends after import. On TensorFlow 2.16+, ``tf.keras`` is Keras 3, so the TensorFlow models also need Keras to run on TensorFlow. To compare backends, use separate processes.

Importing Classes
*******************

When importing classes from DLOmix, you should always use the top-level module:

.. code-block:: python

   # Correct way to import
   from dlomix.models import ChargeStatePredictor
   from dlomix.layers import AttentionLayer

   # The backend implementation (TensorFlow or PyTorch) is selected from DLOMIX_BACKEND

The class names in your code will always be the same (e.g., :code:`ChargeStatePredictor`, :code:`AttentionLayer`), regardless of which backend is being used.

Shared and Backend-Specific Modules
***********************************

- **Models and layers** (``dlomix.models``, ``dlomix.layers``) have one implementation per backend: TensorFlow models subclass ``tf.keras.Model``, PyTorch models subclass ``torch.nn.Module``. The implementation files follow this naming convention:

  - :code:`src/dlomix/{SUBPACKAGE_NAME}/{MODULE_NAME}.py` (for TensorFlow, always with no suffix)
  - :code:`src/dlomix/{SUBPACKAGE_NAME}/{MODULE_NAME}_torch.py` (for PyTorch, always with a :code:`_torch` suffix)

- **Losses and metrics** (``dlomix.losses``, ``dlomix.eval``) are implemented once, against ``keras.ops``, and run on whichever backend is active. They accept and return the active framework's tensors.

- **Data and reports** (``dlomix.data``, ``dlomix.reports``) are backend-agnostic. Datasets produce TensorFlow or PyTorch tensors depending on their ``dataset_type`` (``"tf"`` or ``"pt"``).

Every model is available on both backends:

.. list-table::
   :header-rows: 1

   * - Model
     - TensorFlow/Keras
     - PyTorch
   * - ``PrositRetentionTimePredictor``
     - ✅
     - ✅
   * - ``PrositIntensityPredictor``
     - ✅
     - ✅
   * - ``ChargeStatePredictor``
     - ✅
     - ✅
   * - ``DetectabilityModel``
     - ✅
     - ✅
   * - ``DeepLCRetentionTimePredictor``
     - ✅
     - ✅
   * - ``Ionmob``
     - ✅
     - ✅

The two implementations of each model compute the same function: given the same weights, they produce the same outputs, and they start training from the same weight distributions (the PyTorch models initialize their layers like Keras). The test suite checks both by copying the Keras weights of each model into its PyTorch counterpart.

Notes for PyTorch
*****************

- **Losses return one value per sample**, following the Keras convention. In a hand-written training loop, reduce them before backpropagating:

  .. code-block:: python

     from dlomix.losses import masked_spectral_distance

     loss = masked_spectral_distance(y_true, y_pred).mean()
     loss.backward()

- **Model outputs are probabilities, never logits**, as on TensorFlow. For example, ``ChargeStatePredictor(model_flavour="dominant")`` returns softmax probabilities: train it with ``nn.NLLLoss()`` on ``torch.log(probabilities)``, not with ``nn.CrossEntropyLoss``.

- **Metrics take** ``(y_true, y_pred)``, as on TensorFlow, including ``adjusted_mean_absolute_error`` and ``adjusted_mean_squared_error``.
