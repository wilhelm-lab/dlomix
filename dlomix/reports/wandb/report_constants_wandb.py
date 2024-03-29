TRAIN_SECTION_WANDB = """
The following section shows the different metrics that were used to track the training. All used metrics are added by default. The first subsection shows the metrics per epoch, whereas the second subsection show the metrics per batch."""

VAL_SECTION_WANDB = """
The following section shows the different metrics that were used to track the validation. All used metrics are added by default. The metrics are shown per epoch."""

TRAIN_VAL_SECTION_WANDB = """
The following section shows the training metrics in comparison with the validation metrics. The training loss is a metric used to assess how well a model fits the training data. In contrast the validation loss assesses the performance of the model on previously unseen data. Plotting both curves in the same plot provides a quick way of diagnosing the model for overfitting or underfitting. All used metrics are added by default. The resolution of this section is per epoch."""

SPECTRAL_ANGLE_SECTION_WANDB = """
The spectral angle plot shows the spectral angle between the predicted intensities and the actual intensities faceted by precursor charge. The spectral angle is an evaluation metric for spectra. The closer to 1 the more similar the spectra are. The closer to 0 the more different the spectra are.
"""

DATA_SECTION_WANDB = """
The following section is showing a simple explorative data analysis of the used dataset. The first histogram shows the distribution of peptide lengths in the data set, while the second histogram shows the distribution of indexed retention times.
"""

RESIDUAL_SECTION_WANDB = """
This section shows the residuals histograms. Each plot shows the residuals of each of the compared models
"""

R2_SECTION_WANDB = """
The following plot displays the R2 score for all the compared models.
"""

DENSITY_SECTION_WANDB = """
This section displays the density plots for all compared models.
"""

MODEL_SECTION_WANDB = """
The following section shows information about the model. The table below contains information about the models' layers.
"""

METRICS_TO_EXCLUDE = [
    "epoch/learning_rate",
    "epoch/epoch",
    "batch/learning_rate",
    "batch/batch_step",
]
