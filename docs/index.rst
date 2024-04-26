.. DLOmix documentation master file, created by
   sphinx-quickstart on Thu Sep 30 16:19:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DLOmix - Deep Learning for Proteomics
=====================================

|docs-badge| - |build-badge| - |pypi-badge|

.. |docs-badge| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
    :target: https://dlomix.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |build-badge| image:: https://github.com/wilhelm-lab/dlomix/actions/workflows/build.yaml/badge.svg
    :target: https://github.com/wilhelm-lab/dlomix/actions/workflows/build.yaml
    :alt: Build Status
.. |pypi-badge| image:: https://github.com/wilhelm-lab/dlomix/actions/workflows/pypi.yaml/badge.svg
    :target: https://github.com/wilhelm-lab/dlomix/actions/workflows/pypi.yaml
    :alt: PyPI Status

DLOmix is a deep learning framework for proteomics. It is designed to provide proteomics researchers with high-level functionality for building and training deep learning models for proteomics data.

The goal of DLOmix is to be easy to use and flexible, while still providing the ability to build complex models. DLOmix is built on top of TensorFlow/Keras and maintains compitability with many Keras features.



.. automodule:: dlomix
    :members:

.. include:: notes/installation.rst
.. include:: notes/quickstart.rst


.. toctree::
   :glob:
   :reversed:
   :maxdepth: 2
   :caption: How To

   notes/*


.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   dlomix


Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
