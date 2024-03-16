TRAIN_SECTION_WANDB = """
The following section shows the different metrics that were used to track the training. All used metrics are added by default. This section offers per batch as well as per epoch resolution."""

VAL_SECTION_WANDB = """
The following section shows the different metrics that were used to track the validation. All used metrics are added by default. This section offers per batch as well as per epoch resolution."""

TRAIN_VAL_SECTION_WANDB = """
The following section shows the training metrics in comparision with the validation metrics. The training loss is a metric used to assess how well a model fits the training data. In contrast the validation loss assesses the performance of the model on previously unseen data. Plotting both curves in the same plot provides a quick way of diagnosing the model for overfitting or underfitting. All used metrics are added by default. The resolution of this section is per epoch."""

SPECTRAL_ANGLE_SECTION_WANDB = """
The spectral angle plot shows the spectral angle between the predicted intensities and the actual intensities faceted by precursor charge. The spectral angle is an evaluation metric for spectra. The closer to 1 the more similar the spectra are. The closer to 0 the more different the spectra are.
"""
