"""Evaluate a model, or predictions made elsewhere, on a split of a dataset."""

from typing import List, Optional

import numpy as np
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
            f"{type(model).__name__}. Predict the split yourself and pass the result "
            "to score_intensity_predictions."
        )
    dataset._check_if_split_exists(split)

    # inputs only and never shuffled, so that predictions line up with the rows
    inputs = to_tf_tensor_dataset(
        dataset.hf_dataset[split],
        input_columns=dataset._get_input_tensor_column_names(),
        batch_size=dataset.batch_size,
    )
    return score_intensity_predictions(
        model.predict(inputs, verbose=0),
        dataset,
        split=split,
        precursor_charge_column=precursor_charge_column,
        columns=columns,
    )


def score_intensity_predictions(
    predictions,
    dataset,
    split: str = "test",
    precursor_charge_column: str = "precursor_charge_onehot",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Score predicted spectra of a split against the observed ones.

    The scoring of :func:`evaluate_intensity` for predictions made elsewhere: by an
    :class:`~dlomix.pipelines.InferencePipeline` from the raw rows, by a PyTorch
    model, or by another tool. They are post-processed and scored in the same way,
    so the spectral angles are comparable.

    Parameters
    ----------
    predictions:
        One predicted spectrum per row of the split, in the split's row order:
        an array of shape ``(rows, ions)`` or a sequence of such rows.
    dataset:
        The :class:`~dlomix.data.FragmentIonIntensityDataset` holding the split.
    split:
        The split the predictions belong to: ``"train"``, ``"val"`` or ``"test"``
        (default).
    precursor_charge_column:
        Column with the one-hot precursor charge, needed to mask the fragment
        charges a precursor cannot produce.
    columns:
        Further columns of the split to return, e.g. identifiers. Defaults to the
        dataset's ``dataset_columns_to_keep``.

    Returns
    -------
    pandas.DataFrame
        As :func:`evaluate_intensity`: one row per spectrum, in the order of the
        split, with the post-processed prediction (``intensities_pred``) and the
        ``spectral_angle``.
    """
    dataset._check_if_split_exists(split)
    hf_split = dataset.hf_dataset[split]
    if len(predictions) != len(hf_split):
        raise ValueError(
            f"{len(predictions)} predictions for the {len(hf_split)} rows of the "
            f"'{split}' split: pass one prediction per row, in the split's order."
        )

    sequence_column = SequenceParsingProcessor.PARSED_COL_NAMES["seq"]
    label_column = dataset.label_column[0]
    if columns is None:
        columns = [c for c in dataset.dataset_columns_to_keep if c in hf_split.features]
    selected = [sequence_column, label_column, precursor_charge_column, *columns]
    result = hf_split.select_columns(list(dict.fromkeys(selected))).to_pandas()
    result[PREDICTIONS_COLUMN] = list(np.asarray(predictions))

    return normalize_intensity_predictions(
        result,
        sequence_column_name=sequence_column,
        labels_column_name=label_column,
        predictions_column_name=PREDICTIONS_COLUMN,
        precursor_charge_column_name=precursor_charge_column,
    )
