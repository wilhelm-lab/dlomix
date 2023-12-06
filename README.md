# DLOmix

[![Docs](https://readthedocs.org/projects/dlomix/badge/?version=stable)](https://dlomix.readthedocs.io/en/stable/?badge=stable)
[![Build](https://github.com/wilhelm-lab/dlomix/actions/workflows/build.yaml/badge.svg)](https://github.com/wilhelm-lab/dlomix/actions/workflows/build.yaml)
[![PyPI](https://github.com/wilhelm-lab/dlomix/actions/workflows/pypi.yaml/badge.svg)](https://github.com/wilhelm-lab/dlomix/actions/workflows/pypi.yaml)

**DLOmix** is a Python framework for Deep Learning in Proteomics. Initially built on top of TensorFlow/Keras, support for PyTorch can however be integrated once the main API is established.

## Usage
Experiment a simple retention time prediction use-case using Google Colab &nbsp;&nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wilhelm-lab/dlomix/blob/develop/notebooks/Example_RTModel_Walkthrough_colab.ipynb)

A version that includes experiment tracking with [Weights and Biases](https://www.wandb.ai) is available here &nbsp;&nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wilhelm-lab/dlomix/blob/develop/notebooks/Example_RTModel_Walkthrough_colab-weights-and-biases.ipynb)

**Resources Repository**

More learning resources can be found in the [dlomix-resources](https://github.com/wilhelm-lab/dlomix-resources) repository.

## Installation

Run the following to install:
```bash
$ pip install dlomix
```

If you would like to use [Weights & Biases](wandb.ai) for experiment tracking and use the available reports for Retention Time under `/notebooks`, please install the optional `wandb` python dependency with `dlomix` by running:

```bash
$ pip install dlomix[wandb]
```

**General Overview**
-  `data`: structures for modeling the input data, currently based on `tf.Dataset`
-  `eval`: classes for evaluating models and reporting results
-  `layers`: custom layers used for building models, based on `tf.keras.layers.Layer`
-  `losses`: custom losses to be used for training with `model.fit()`
- `models`: common model architectures for the relevant use-cases based on `tf.keras.Model` to allow for using the Keras training API
-  `pipelines`: an exemplary high-level pipeline implementation
-  `reports`: classes for generating reports related to the different tasks
-  `constants.py`: constants and configuration values
-  `utils.py`: utility functions



**Use-cases**

- Retention Time Prediction: 
    - a regression problem where the retention time of a peptide sequence is to be predicted. 



**To-Do**

Functionality:
- [X] integrate prosit
- [ ] extend pipeline for different types of models and backbones
- [ ] extend pipeline to allow for fine-tuning with custom datasets
- [X] add residual plots to reporting, possibly other regression analysis tools
- [X] output reporting results as PDF
- [ ] extend data representation to include modifications
- [X] refactor reporting module to use W&B Report API (Retention Time)

Package structure:

- [X] integrate `deeplc.py` into `models.py`, preferably introduce a package structure (e.g. `models.retention_time`)
- [X] add references for implemented models in the ReadMe
- [ ] introduce a style guide and checking (e.g. PEP)
- [X] plan documentation (sphinx and readthedocs)


## Developing DLOmix
To install dlomix, along with the tools needed to develop and run tests, run the following command in your virtualenv:
```bash
$ pip install -e .[dev]
```


**References:**

[**Prosit**]

[1] Gessulat, S., Schmidt, T., Zolg, D. P., Samaras, P., Schnatbaum, K., Zerweck, J., ... & Wilhelm, M. (2019). Prosit: proteome-wide prediction of peptide tandem mass spectra by deep learning. Nature methods, 16(6), 509-518.

[**DeepLC**]

[2] DeepLC can predict retention times for peptides that carry as-yet unseen modifications
Robbin Bouwmeester, Ralf Gabriels, Niels Hulstaert, Lennart Martens, Sven Degroeve
bioRxiv 2020.03.28.013003; doi: 10.1101/2020.03.28.013003

[3] Bouwmeester, R., Gabriels, R., Hulstaert, N. et al. DeepLC can predict retention times for peptides that carry as-yet unseen modifications. Nat Methods 18, 1363–1369 (2021). https://doi.org/10.1038/s41592-021-01301-5
 
