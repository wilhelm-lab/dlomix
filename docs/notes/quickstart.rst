Quickstart
**********

Experiment a simple retention time prediction use-case using Google Colab |Colab RT|

.. |Colab RT| image:: https://colab.research.google.com/assets/colab-badge.svg
       :target: https://colab.research.google.com/github/wilhelm-lab/dlomix/blob/main/notebooks/Example_RTModel_Walkthrough_colab.ipynb

A version that includes experiment tracking with `Weights and Biases`_ is available here |Colab RTwandb|

.. _Weights and Biases: https://wandb.ai/

.. |Colab RTwandb| image:: https://colab.research.google.com/assets/colab-badge.svg
       :target: https://colab.research.google.com/github/wilhelm-lab/dlomix/blob/main/notebooks/Example_RTModel_Walkthrough_colab-weights-and-biases.ipynb

More example notebooks:

- Fragment ion intensity prediction (Prosit) |Colab Intensity|
- Precursor charge state prediction, TensorFlow |Colab PCSP| and PyTorch |Colab PCSP torch|
- Retention time prediction with DeepLC |Colab DeepLC|

.. |Colab Intensity| image:: https://colab.research.google.com/assets/colab-badge.svg
       :target: https://colab.research.google.com/github/wilhelm-lab/dlomix/blob/main/notebooks/Example_IntensityModel_Walkthrough_colab.ipynb

.. |Colab PCSP| image:: https://colab.research.google.com/assets/colab-badge.svg
       :target: https://colab.research.google.com/github/wilhelm-lab/dlomix/blob/main/notebooks/Example_PCSP_huggingface_walkthrough.ipynb

.. |Colab PCSP torch| image:: https://colab.research.google.com/assets/colab-badge.svg
       :target: https://colab.research.google.com/github/wilhelm-lab/dlomix/blob/main/notebooks/Example_PCSP_huggingface_walkthrough_torch.ipynb

.. |Colab DeepLC| image:: https://colab.research.google.com/assets/colab-badge.svg
       :target: https://colab.research.google.com/github/wilhelm-lab/dlomix/blob/main/notebooks/Example_DeepLC_RetentionTimePrediction.ipynb

All notebooks are in the `notebooks folder <https://github.com/wilhelm-lab/dlomix/tree/main/notebooks>`_ of the repository.

From a dataset to a scored model
================================

A model takes the arguments that must agree with its dataset from the dataset
(``from_dataset`` exists on every model), and a fragment ion intensity model is scored
on a split in one call, with the predictions post-processed as Prosit does:

.. code-block:: python

   from dlomix.data import FragmentIonIntensityDataset
   from dlomix.models import PrositIntensityPredictor
   from dlomix.reports import evaluate_intensity

   dataset = FragmentIonIntensityDataset(
       data_source="train.parquet",
       test_data_source="test.parquet",
       model_features=["precursor_charge_onehot", "collision_energy_aligned_normed"],
   )
   model = PrositIntensityPredictor.from_dataset(dataset)
   # ... model.compile(...), model.fit(dataset.tensor_train_data, ...)
   scores = evaluate_intensity(model, dataset)  # one row per spectrum
   scores["spectral_angle"].median()

Predictions made elsewhere (another tool, another backend) are scored the same way
with ``dlomix.reports.score_intensity_predictions(predictions, dataset)``.

Fine-tuning a published model on new modifications
==================================================

``FineTunePipeline.from_pretrained`` (TensorFlow backend) takes a saved or published
inference bundle and Hugging Face data. It builds the dataset with the bundle's
preprocessing settings, finds the tokens that are new to the model and adds them to
its vocabulary:

.. code-block:: python

   from dlomix.pipelines import FineTunePipeline

   pipeline = FineTunePipeline.from_pretrained(
       "my-org/prosit-intensity",  # Hub repo id, saved path or InferencePipeline
       data,  # a Hugging Face Dataset or DatasetDict
       initialization_strategy="best-fit",
       trainable="new_tokens",  # or "embedding", or "all" (default)
   )
   history = pipeline.finetune()

Every step is public and can be done by hand instead: ``find_new_tokens`` and
``extend_alphabet`` (``dlomix.data``), ``adapt_pretrained_model`` (``dlomix.models``)
and ``FineTunePipeline.from_dataset_and_model``. See the module documentation of
``dlomix.pipelines.finetune``.

.. tip::
    More learning resources can be found in the dlomix-resources_ GitHub repository.
.. _dlomix-resources: https://github.com/wilhelm-lab/dlomix-resources/
