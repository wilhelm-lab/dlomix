MODEL_SECTION = """
The following section shows information about the model. The table below contains information about the models' layers."""

DATA_SECTION = """
The following section is showing a simple explorative data analysis of the used dataset. The first histogram shows the
distribution of peptide lengths in the data set, while the second histogram shows the distribution of indexed retention
times. The first density plot shows the density of retention time per peptide length. The second density plot depicts the
density of Levenshtein distances per petide length."""

TRAIN_SECTION = """
The following section shows the different metrics that were used to track the training. All used metrics are added by
default. The resolution of this section is per epoch."""

VAL_SECTION = """
The following section shows the different metrics that were used to track the validation. All used metrics are added by
default. The resolution of this section is per epoch."""

TRAIN_VAL_SECTION = """
The following section shows the training metrics in comparision with the validation metrics. The training loss is a metric used to assess how well a model fits the training data. In contrast the validation
loss assesses the performance of the model on previously unseen data. Plotting both curves in the same plot provides a
quick way of diagnosing the model for overfitting or underfitting. All used metrics are added by default."""

RESIDUALS_SECTION = """
This section shows a histogram of the residuals of the model. Residuals are the difference between the actual values and
the predicted values of the test set. A histogram of those residuals offers a way of assessing the models performance."""

DENSITY_SECTION = """
This section shows the density plot. A better explanation of the density plot is yet to be done."""

R2_SECTION = """
The R2 of given predictions is R2_SCORE_VALUE. The R2 score is a metric that is calculated using scikit learns's function
r2_score. It evaluates the performance of the model on the test set and compares the predicted values with the actual values.
The best possible score is 1.0. The lower the score the worse the model's prediction."""
