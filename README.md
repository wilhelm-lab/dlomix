# DLOmix

**DLOmix** is a python framework for Deep Learning in Proteomics. Initially built ontop of TensorFlow/Keras, support for PyTorch can however be integrated once the main API is established.

Experiment a simple retention time prediction use-case:
- using Google Colab &nbsp;&nbsp; [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wilhelm-lab/dlomix/blob/develop/notebooks/Example_RTModel_Walkthrough_colab.ipynb)



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
- [ ] add residual plots to reporting, possibly other regression analysis tools
- [ ] output reporting results as PDF
- [ ] extend data representation to include modifications

Package structure:

- [X] integrate `deeplc.py` into `models.py`, preferably introduce a package structure (e.g. `models.retention_time`)
- [X] add references for implemented models in the ReadMe
- [ ] introduce a style guide and checking (e.g. PEP)
- [ ] plan documentation (sphinx and readthedocs)


 


**References:**

[Prosit]
Gessulat, S., Schmidt, T., Zolg, D. P., Samaras, P., Schnatbaum, K., Zerweck, J., ... & Wilhelm, M. (2019). Prosit: proteome-wide prediction of peptide tandem mass spectra by deep learning. Nature methods, 16(6), 509-518.

[DeepLC]
DeepLC can predict retention times for peptides that carry as-yet unseen modifications
Robbin Bouwmeester, Ralf Gabriels, Niels Hulstaert, Lennart Martens, Sven Degroeve
bioRxiv 2020.03.28.013003; doi: 10.1101/2020.03.28.013003
 
