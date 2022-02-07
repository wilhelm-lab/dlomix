# DLOmix

[![Docs](https://readthedocs.org/projects/docs/badge/?version=latest)](https://dlomix.readthedocs.io/en/latest/?badge=latest)
[![Build](https://github.com/wilhelm-lab/dlomix/actions/workflows/build.yaml/badge.svg)](https://github.com/wilhelm-lab/dlomix/actions/workflows/build.yaml)
[![PyPI](https://github.com/wilhelm-lab/dlomix/actions/workflows/pypi.yaml/badge.svg)](https://github.com/wilhelm-lab/dlomix/actions/workflows/pypi.yaml)

**DLOmix** is a python framework for Deep Learning in Proteomics. Initially built ontop of TensorFlow/Keras, support for PyTorch can however be integrated once the main API is established.

## Quickstart with [Docker](https://docs.docker.com/get-docker/) 
```bash
docker pull animesh1977/dlomix
docker run -it --rm animesh1977/dlomix bash
curl https://raw.githubusercontent.com/animesh/dlomix/master/checkDLomix.py > checkDLomix.py
curl https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/proteomTools_train_val.csv > train.csv
curl https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/proteomTools_test.csv > test.csv
python checkDLomix.py train.csv test.csv

```

#### Building the Docker image locally:
- download the repo `git clone http://github.com/animesh/dlomix`
- move into the cloned directory: `cd dlomix`
- create image with CMD:  `docker build --no-cache .` (please note the "." in the end)


## Usage
Experiment a simple retention time prediction use-case using Google Colab &nbsp;&nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wilhelm-lab/dlomix/blob/develop/notebooks/Example_RTModel_Walkthrough_colab.ipynb)

**Resources Repository:**

More learning resources can be found in the [dlomix-resources](https://github.com/wilhelm-lab/dlomix-resources) repository.

## Installation
Run the following to install:
```bash
$ pip install dlomix
``` 

**General Overview:**
- `data.py`: structures for modelling the input data, currently based on `tf.Dataset`
- `models.py`: common model architectures for the relevant use-cases based on `tf.keras.Model` to allow for using the Keras training API
- `pipeline.py`: an exemplary high-level pipeline implementation
-  `eval.py`: classes for evaluating models and reporting results
-  `eval_utils.py`: custom evaluation metrics implemented in TensorFlow/Keras
-  `constants.py`: constants and configuration values needs for the `pipeline` class.



**Use-cases:**

- Retention Time Prediction: 
    - a regression problem where the the retention time of a peptide sequence is to be predicted. 



**To-Do:**

Functionality:
- [X] integrate prosit
- [ ] extend pipeline for different types of models and backbones
- [ ] extend pipeline to allow for fine-tuning with custom datasets
- [X] add residual plots to reporting, possibly other regression analysis tools
- [ ] output reporting results as PDF
- [ ] extend data representation to include modifications

Package structure:

- [X] integrate `deeplc.py` into `models.py`, preferably introduce a package structure (e.g. `models.retention_time`)
- [X] add references for implemented models in the ReadMe
- [ ] introduce a style guide and checking (e.g. PEP)
- [ ] plan documentation (sphinx and readthedocs)


## Developing DLOmix
To install dlomix, along with the the tools needed to develop and run tests, run the following command in your virtualenv:
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
 
