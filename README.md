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





 