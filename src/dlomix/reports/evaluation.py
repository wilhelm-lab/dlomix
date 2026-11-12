"""Evaluate a model on a split of a dataset."""

from typing import List, Optional

import pandas as pd

from ..data.processing.processors import SequenceParsingProcessor
from ..data.tensor_conversion import to_tf_tensor_dataset
from .postprocessing import normalize_intensity_predictions

PREDICTIONS_COLUMN = "intensities_pred"


def evaluate_intensity(
    model,
    dataset,
    split: str = "test",
    precursor_charge_column: str = "precursor_charge_onehot",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Predict the spectra of a split and score them against the observed ones.

    The predictions are post-processed as Prosit does (see
    :func:`~dlomix.reports.postprocessing.normalize_intensity_predictions`): negative
    intensities are clipped, the ions a peptide cannot produce are set to -1, and
    each spectrum is normalized to its base peak. The spectral angle is computed
    between the post-processed prediction and the observed intensities.

    Parameters
    ----------
    model:
        A Keras fragment ion intensity model, e.g.
        :class:`~dlomix.models.PrositIntensityPredictor`.
    dataset:
        The :class:`~dlomix.data.FragmentIonIntensityDataset` the model's inputs come
        from.
    split:
        The split to evaluate: ``"train"``, ``"val"`` or ``"test"`` (default).
    precursor_charge_column:
        Column with the one-hot precursor charge, needed to mask the fragment
        charges a precursor cannot produce.
    columns:
        Further columns of the split to return, e.g. identifiers. Defaults to the
        dataset's ``dataset_columns_to_keep``.

    Returns
    -------
    pandas.DataFrame
        One row per spectrum, in the order of the split (also for a shuffled
        training split), with the parsed sequence, the observed intensities, the
        precursor charge, ``columns``, the post-processed prediction
        (``intensities_pred``) and the ``spectral_angle``.
    """
    if not hasattr(model, "predict"):
        raise TypeError(
            "evaluate_intensity needs a Keras model (with a predict method), got "
            f"{type(model).__name__}."
        )
    dataset._check_if_split_exists(split)
    hf_split = dataset.hf_dataset[split]

    sequence_column = SequenceParsingProcessor.PARSED_COL_NAMES["seq"]
    label_column = dataset.label_column[0]
    if columns is None:
        columns = [c for c in dataset.dataset_columns_to_keep if c in hf_split.features]
    selected = [sequence_column, label_column, precursor_charge_column, *columns]
    result = hf_split.select_columns(list(dict.fromkeys(selected))).to_pandas()

    # inputs only and never shuffled, so that predictions line up with the rows
    inputs = to_tf_tensor_dataset(
        hf_split,
        input_columns=dataset._get_input_tensor_column_names(),
        batch_size=dataset.batch_size,
    )
    result[PREDICTIONS_COLUMN] = list(model.predict(inputs, verbose=0))

    return normalize_intensity_predictions(
        result,
        sequence_column_name=sequence_column,
        labels_column_name=label_column,
        predictions_column_name=PREDICTIONS_COLUMN,
        precursor_charge_column_name=precursor_charge_column,
    )
