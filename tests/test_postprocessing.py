"""Prosit-style post-processing of predicted fragment ion intensities.

The 174 intensities of a spectrum are laid out as 29 positions x 2 ion types (y, b)
x 3 fragment charges. Post-processing masks (-1) the ions a peptide cannot produce
and normalizes each spectrum to its base peak.
"""

import numpy as np
import pandas as pd
import pytest

from dlomix.reports.postprocessing import normalize_intensity_predictions

# parsed peptides (lists of residues, as SequenceParsingProcessor produces them)
PEPTIDES = [
    ["L", "F", "G", "N", "K", "N", "C[UNIMOD:4]", "V", "T", "I", "H", "Q", "R"],
    ["G", "L", "T", "D", "K", "L", "T", "Q", "A"],
]
CHARGES = [2, 3]


def _frame(predictions, labels=None):
    data = {
        "sequences": PEPTIDES,
        "intensities_pred": [list(p) for p in predictions],
        "precursor_charge_onehot": [list(np.eye(6, dtype=int)[c - 1]) for c in CHARGES],
    }
    if labels is not None:
        data["intensities_raw"] = [list(label) for label in labels]
    return pd.DataFrame(data)


def _possible_ions(length, charge):
    """Mask of the ions a peptide can produce, in the 174 layout."""
    possible = np.zeros((29, 2, 3), dtype=bool)
    possible[: length - 1, :, : min(charge, 3)] = True
    return possible.ravel()


def test_masks_the_ions_a_peptide_cannot_produce():
    predictions = np.random.default_rng(0).random((2, 174)) + 0.1
    result = normalize_intensity_predictions(
        _frame(predictions), compute_spectral_angle=False
    )
    for peptide, charge, processed in zip(
        PEPTIDES, CHARGES, result["intensities_pred"]
    ):
        processed = np.asarray(processed)
        possible = _possible_ions(len(peptide), charge)
        assert np.all(processed[~possible] == -1)
        assert np.all(processed[possible] >= 0)
    # 12 positions x 2 ion types x 2 charges, and 8 x 2 x 3
    assert [int((np.asarray(p) >= 0).sum()) for p in result["intensities_pred"]] == [
        48,
        48,
    ]


def test_clips_negatives_and_normalizes_to_the_base_peak():
    predictions = np.full((2, 174), 0.5)
    predictions[:, 0] = 2.0  # base peak (y1, charge 1)
    predictions[:, 1] = -0.3  # negative intensity (y1, charge 2)
    result = normalize_intensity_predictions(
        _frame(predictions), compute_spectral_angle=False
    )
    for processed in result["intensities_pred"]:
        processed = np.asarray(processed)
        assert processed[0] == pytest.approx(1.0)
        assert processed[1] == 0.0
        assert processed[processed >= 0].max() == pytest.approx(1.0)


def test_spectral_angle_of_a_perfect_prediction_is_one():
    labels = np.random.default_rng(1).random((2, 174))
    for i, (peptide, charge) in enumerate(zip(PEPTIDES, CHARGES)):
        possible = _possible_ions(len(peptide), charge)
        labels[i] /= labels[i, possible].max()  # base peak among the possible ions
        labels[i, ~possible] = -1
    result = normalize_intensity_predictions(_frame(labels, labels=labels))
    np.testing.assert_allclose(
        np.stack(result["intensities_pred"].to_numpy()), labels, atol=1e-6
    )
    # Prosit's masked SA keeps identical spectra just below 1: its epsilon rescales
    # each intensity slightly differently, which arccos magnifies near 1 (~3e-4)
    np.testing.assert_allclose(result["spectral_angle"], 1.0, atol=1e-3)


RAW_PEPTIDES = ["[]-LFGNKNC[UNIMOD:4]VTIHQR-[]", "[UNIMOD:1]-GLTDKLTQA-[]"]


def test_raw_sequence_strings_are_counted_in_residues():
    # a raw string must give the same masks as its parsed residues, not count
    # characters ("[]-LFGNKNC[UNIMOD:4]VTIHQR-[]" has 29 characters, 13 residues)
    predictions = np.random.default_rng(2).random((2, 174)) + 0.1
    parsed = normalize_intensity_predictions(
        _frame(predictions), compute_spectral_angle=False
    )
    raw = _frame(predictions)
    raw["sequences"] = RAW_PEPTIDES
    raw = normalize_intensity_predictions(raw, compute_spectral_angle=False)
    np.testing.assert_array_equal(
        np.stack(raw["intensities_pred"].to_numpy()),
        np.stack(parsed["intensities_pred"].to_numpy()),
    )


def test_rows_are_matched_by_position_not_index_label():
    # a filtered or reordered DataFrame keeps its index labels; lengths must still
    # go to the right rows
    predictions = np.random.default_rng(3).random((2, 174)) + 0.1
    expected = normalize_intensity_predictions(
        _frame(predictions), compute_spectral_angle=False
    )
    shuffled = _frame(predictions).iloc[::-1]  # index [1, 0]
    result = normalize_intensity_predictions(shuffled, compute_spectral_angle=False)
    np.testing.assert_array_equal(
        np.stack(result["intensities_pred"].to_numpy()),
        np.stack(expected["intensities_pred"].to_numpy())[::-1],
    )
    filtered = _frame(predictions).iloc[[1]]  # index [1] only
    result = normalize_intensity_predictions(filtered, compute_spectral_angle=False)
    np.testing.assert_array_equal(
        np.asarray(result["intensities_pred"].iloc[0]),
        np.asarray(expected["intensities_pred"].iloc[1]),
    )


def test_integer_encoded_sequences_are_rejected():
    data = _frame(np.random.default_rng(4).random((2, 174)))
    data["sequences"] = [[21, 5, 7, 0, 0], [21, 3, 0, 0, 0]]
    with pytest.raises(ValueError, match="integer-encoded"):
        normalize_intensity_predictions(data, compute_spectral_angle=False)
