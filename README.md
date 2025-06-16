# DLOmix

[![Docs](https://readthedocs.org/projects/dlomix/badge/?version=stable)](https://dlomix.readthedocs.io/en/stable/?badge=stable)
[![Build](https://github.com/wilhelm-lab/dlomix/actions/workflows/build.yaml/badge.svg)](https://github.com/wilhelm-lab/dlomix/actions/workflows/build.yaml)
[![PyPI](https://github.com/wilhelm-lab/dlomix/actions/workflows/pypi.yaml/badge.svg)](https://github.com/wilhelm-lab/dlomix/actions/workflows/pypi.yaml)

**DLOmix** is a Python framework for Deep Learning in Proteomics. DLOmix provides multi-backend support for both **TensorFlow/Keras** and **PyTorch**, allowing researchers to choose their preferred deep learning framework while maintaining identical APIs and functionality. The dataset module is built upon HuggingFace `datasets` and can provide both TensorFlow and PyTorch tensors.

**Note:Multi-backend support was introduced in `dlomix==0.2`. Earlier versions supported TensorFlow/Keras only. **

The PyTorch implementation was largely introduced during a hackathon as part of the EuBIC Developer Meeting 2025. We appreciate the efforts and contributions of the team who joined the hackathon and the efforts of the EuBIC team and organizers.

**

## Backend Selection

DLOmix automatically detects and uses the appropriate backend based on your environment setup. You can control which backend to use through the `DLOMIX_BACKEND` environment variable:

### TensorFlow Backend (Default)
```bash
# Set TensorFlow as backend (default)
export DLOMIX_BACKEND=tensorflow
# or
export DLOMIX_BACKEND=tf

# Install DLOmix with TensorFlow
pip install dlomix[tensorflow]

# Or install tensorflow separately (existing installation), then only install dlomix
pip install dlomix
```

### PyTorch Backend
```bash
# Set PyTorch as backend
export DLOMIX_BACKEND=pytorch
# or
export DLOMIX_BACKEND=torch
# or
export DLOMIX_BACKEND=pt

# Install DLOmix with PyTorch support
pip install dlomix[pytorch]

# Or install pytorch separately (existing installation), then only install dlomix
pip install dlomix
```

**Note**: The backend must be set **before** importing DLOmix. If no backend is specified, DLOmix defaults to TensorFlow with a user warning.

## Usage
Experiment a simple retention time prediction use-case using Google Colab &nbsp;&nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wilhelm-lab/dlomix/blob/develop/notebooks/Example_RTModel_Walkthrough_colab.ipynb)

A version that includes experiment tracking with [Weights and Biases](https://www.wandb.ai) is available here &nbsp;&nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wilhelm-lab/dlomix/blob/develop/notebooks/Example_RTModel_Walkthrough_colab-weights-and-biases.ipynb)

**Resources Repository**

More learning resources can be found in the [dlomix-resources](https://github.com/wilhelm-lab/dlomix-resources) repository.

## Installation

### Quick Start
```bash
# Basic installation
# TensorFlow and PyTorch not installed, please install separately
pip install dlomix

# Install DLOmix and additionally install specific backend
pip install dlomix[tensorflow]  # TensorFlow backend
pip install dlomix[pytorch]     # PyTorch backend

```


**General Package Overview**

DLOmix provides a unified API across both TensorFlow and PyTorch backends:

-  `data`: structures for modeling input data, processing functions, and feature extractions based on Hugging Face datasets `Dataset` and `DatasetDict` (backend-agnostic)
-  `eval`: classes for evaluating models and reporting results (backend-specific implementations)
-  `layers`: custom layers for building models
   - TensorFlow: based on `tf.keras.layers.Layer`
   - PyTorch: based on `torch.nn.Module`
-  `losses`: custom loss functions
   - TensorFlow: compatible with `model.fit()`
   - PyTorch: compatible with standard PyTorch training loops
- `models`: common model architectures for relevant use-cases
   - TensorFlow: based on `tf.keras.Model`
   - PyTorch: based on `torch.nn.Module`
-  `pipelines`: high-level pipeline implementations (backend-agnostic)
-  `reports`: classes for generating reports (backend-agnostic)
-  `constants.py`: constants and configuration values

**Available Models by Backend**

| Model | TensorFlow/Keras | PyTorch |
|-------|------------|---------|
| `PrositRetentionTimePredictor` [1] | ✅ | ✅ |
| `PrositIntensityPredictor` [1] | ✅ | ✅ |
| `ChargeStatePredictor` | ✅ | ✅ |
| `DetectabilityModel` [4] | ✅ | ✅ |
| `DeepLCRetentionTimePredictor` [2,3] | ✅ | ❌ |
| `Ionmob` [5] | ❌ | ✅ |
| `PIMMS-CF` [6] | ❌ | ⚠ (experimental) |



**Use-cases**

- Retention Time Prediction:
    - a regression problem where the retention time of a peptide sequence is to be predicted.

- Fragment Ion Intensity Prediction:
    - a multi-output regression problem where the intensity values for fragment ions are predicted given a peptide sequence along with some additional features.

- Peptide Detectability (Pfly) [4]:
    - a multi-class classification problem where the detectability of a peptide is predicted given the peptide sequence.


## Developing DLOmix
To install dlomix, along with the tools needed to develop and run tests, run the following command in your virtualenv:
```bash
$ pip install -e .[dev]
```


## References

[**Prosit**]

[1] Gessulat, S., Schmidt, T., Zolg, D. P., Samaras, P., Schnatbaum, K., Zerweck, J., ... & Wilhelm, M. (2019). Prosit: proteome-wide prediction of peptide tandem mass spectra by deep learning. Nature methods, 16(6), 509-518.

[**DeepLC**]

[2] DeepLC can predict retention times for peptides that carry as-yet unseen modifications
Robbin Bouwmeester, Ralf Gabriels, Niels Hulstaert, Lennart Martens, Sven Degroeve
bioRxiv 2020.03.28.013003; doi: 10.1101/2020.03.28.013003

[3] Bouwmeester, R., Gabriels, R., Hulstaert, N. et al. DeepLC can predict retention times for peptides that carry as-yet unseen modifications. Nat Methods 18, 1363–1369 (2021). https://doi.org/10.1038/s41592-021-01301-5

[**Detectability - Pfly**]

[4] Abdul-Khalek, N., Picciani, M., Wimmer, R., Overgaard, M. T., Wilhelm, M., & Gregersen Echers, S. (2024). To fly, or not to fly, that is the question: A deep learning model for peptide detectability prediction in mass spectrometry. bioRxiv, 2024-10.

[**IonMob**]

[5] Teschner, D., Gomez-Zepeda, D., Declercq, A., Łącki, M. K., Avci, S., Bob, K., ... & Hildebrandt, A. (2023). Ionmob: a Python package for prediction of peptide collisional cross-section values. Bioinformatics, 39(9), btad486.

[**PIMMS**]

[6] Webel, H., Niu, L., Nielsen, A.B. et al.
Imputation of label-free quantitative mass spectrometry-based proteomics data using self-supervised deep learning.
Nat Commun 15, 5405 (2024).
https://doi.org/10.1038/s41467-024-48711-5


### Credit

**PyTorch Implementation Hackathon during EuBIC Developer Meeting 2025**

- Ayla Schröder
- Henry Webel
- David Teschner
- Stan Reinders
