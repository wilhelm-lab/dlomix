# dlpro

**dlpro** is a python framework for Deep Learning in Proteomics. Initially built ontop of TensorFlow/Keras, support for PyTorch can however be integrated once the main API is established.

**General Overview:**
- `data.py`: structures for modelling the input data, currently based on `TensorFlow.Dataset`
- `models.py`: common model architectures for the relevant use-cases
- `pipeline.py`: an exemplary high-level pipeline implementation
-  `eval.py`: classes for evaluating models and reporting results
-  `eval_utils.py`: custom evaluation metrics implemented in TensorFlow/Keras
-  `constants.py`: configuration values needs for the `pipeline` class.



**Use-cases:**

- Retention Time Prediction: 
    - a regression problem where the the retention time of a peptide sequence is to be predicted. 



**To-Do:**

Functionality:
- implement deepRT and prosit
- extend pipeline for different types of models and backbones
- extend pipeline to allow for fine-tuning with custom datasets
- add residual plots to reporting, possibly other regression analysis tools
- output reporting results as PDF
- extend data representation to include modifications (based on discussions with Küster Lehrstuhl)

Package structure:
- Adopt a convention for private attributes of objects (datasets, models, etc...)
- introduce a style guide (e.g. PEP)
- integrate `deeplc.py` into `models.py`, preferably introduce a package structure (e.g. `models.retention_time`)
- plan documentation (check sphinx and readthedocs)
- add references for implemented models in the ReadMe
 



 