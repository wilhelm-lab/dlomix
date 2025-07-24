Installation
************

.. caution::


  DLOmix is in its early stages, current release is a pre-release version.
  Changes to the API are highly probable. If you have feedback, ideas for improvements, or if you find a bug, please open an issue on GitHub.

DLOmix can be installed via pip, this installs the package and the main dependencies only, excluding the backend framework to be used (TensorFlow/Keras or PyTorch):

.. code-block:: bash

  pip install dlomix

Backend Selection
******************

DLOmix supports multiple deep learning backends. To use TensorFlow/Keras or PyTorch as a backend together with DLOmix, install with the respective command:

.. code-block:: bash

  pip install dlomix[tensorflow]  # shorter alternatives: [tf]

  pip install dlomix[pytorch]  # shorter alternatives: [torch], [pt]

.. note::

   While DLOmix supports both TensorFlow and PyTorch backends, you only need to install the backend you intend to use. The library will use the available backend automatically. If both backends are installed, TensorFlow will be used by default unless explicitly configured.

.. include:: backend_usage.rst


Development Installation
************************

To get the develop version, you can install directly from GitHub (develop branch):

.. code-block:: bash

  pip install git+https://github.com/wilhelm-lab/dlomix.git@develop

Optional Dependencies
*********************

If you decide to use Weights & Biases for reporting, you can use the extra install command:

.. code-block:: bash

  pip install dlomix[wandb]

For development purposes, you can install all dependencies including both backends:

.. code-block:: bash

  pip install dlomix[dev]
