import logging

import numpy as np
import pytest
from datasets import Dataset

from dlomix.data.processing.feature_extractors import (
    FEATURE_EXTRACTORS_PARAMETERS,
    LookupFeatureExtractor,
    MissingModifiedAminoAcidError,
    MissingModifiedAminoAcidWarning,
)
from dlomix.data.processing.feature_tables import PTM_GAIN_LOOKUP, PTM_LOSS_LOOKUP
from dlomix.data.processing.pipeline import PipelineContext

logger = logging.getLogger(__name__)


def test_lookup_feature_extractor_exact_length(lookup_table):
    # Create a sequence of indices to look up
    sequence_for_lookup = [0, 1, 3]
    sequence_length = len(sequence_for_lookup)
    max_length = 3
    default_value = [-1.0, -1.0]

    # Create the feature extractor
    feature_extractor = LookupFeatureExtractor(
        sequence_column_name="sequence",
        feature_column_name="feature",
        lookup_table=lookup_table,
        feature_default_value=default_value,
        max_length=max_length,
    )

    # Extract features for the given sequence
    feature = feature_extractor._extract_feature(sequence_for_lookup)
    logger.info("Extracted feature:\n%s", feature)

    assert feature.shape == (
        max_length,
        2,
    ), f"Expected feature shape to be ({max_length}, 2), but got {feature.shape}"

    assert np.array_equal(
        feature[:sequence_length],
        np.array([lookup_table[idx] for idx in sequence_for_lookup]),
    ), "The extracted feature values do not match the expected values from the lookup table."

    logger.info(feature[sequence_length:])
    assert (
        len(feature[sequence_length:]) == 0
    ), "The padded feature values do not match the expected default value from the lookup table."


def test_lookup_feature_extractor_with_padding(lookup_table):
    # Create a sequence of indices to look up
    sequence_for_lookup = [0, 1, 3]
    sequence_length = len(sequence_for_lookup)
    max_length = 6
    default_value = [-5.0, -5.0]

    # Create the feature extractor
    feature_extractor = LookupFeatureExtractor(
        sequence_column_name="sequence",
        feature_column_name="feature",
        lookup_table=lookup_table,
        feature_default_value=default_value,
        max_length=max_length,
    )

    # Extract features for the given sequence
    feature = feature_extractor._extract_feature(sequence_for_lookup)
    logger.info("Extracted feature:\n%s", feature)

    assert feature.shape == (
        max_length,
        2,
    ), f"Expected feature shape to be ({max_length}, 2), but got {feature.shape}"

    assert np.array_equal(
        feature[:sequence_length],
        np.array([lookup_table[idx] for idx in sequence_for_lookup]),
    ), "The extracted feature values do not match the expected values from the lookup table."

    assert np.array_equal(
        feature[sequence_length:],
        np.array([default_value] * (max_length - sequence_length)),
    ), "The padded feature values do not match the expected default value from the lookup table."


def test_lookup_feature_extractor_fail_on_missing_modified_amino_acid_raises():
    # negative case: a modified amino acid missing from the lookup table must raise
    # when fail_on_missing_modified_amino_acid=True
    lookup_table = {"M[UNIMOD:35]": [9.0, 9.0]}
    feature_extractor = LookupFeatureExtractor(
        sequence_column_name="sequence",
        feature_column_name="feature",
        lookup_table=lookup_table,
        feature_default_value=[-1.0, -1.0],
        max_length=5,
        fail_on_missing_modified_amino_acid=True,
    )

    sequence_for_lookup = ["[]-", "A", "C[UNIMOD:4]", "-[]"]

    with pytest.raises(MissingModifiedAminoAcidError):
        feature_extractor._extract_feature(sequence_for_lookup)


def test_lookup_feature_extractor_resolves_swapped_double_modification_order():
    # positive case: a doubly-modified token with its two UNIMOD tags in the "wrong"
    # order still resolves to the entry stored under the other order, no default/warn/raise
    lookup_table = {"R[UNIMOD:643][UNIMOD:35]": [9.0, 9.0]}
    feature_extractor = LookupFeatureExtractor(
        sequence_column_name="sequence",
        feature_column_name="feature",
        lookup_table=lookup_table,
        feature_default_value=[-1.0, -1.0],
        max_length=1,
        fail_on_missing_modified_amino_acid=True,
    )

    feature = feature_extractor._extract_feature(["R[UNIMOD:35][UNIMOD:643]"])

    assert np.array_equal(
        feature, np.array([[9.0, 9.0]])
    ), "A swapped tag order should resolve to the same entry as the canonical key."


def test_lookup_feature_extractor_swap_retry_does_not_mask_genuine_misses():
    # negative case: a doubly-modified token that is missing under both orderings must
    # still raise/default/warn as normal — the swap retry isn't a second default
    lookup_table = {"R[UNIMOD:643][UNIMOD:35]": [9.0, 9.0]}
    feature_extractor = LookupFeatureExtractor(
        sequence_column_name="sequence",
        feature_column_name="feature",
        lookup_table=lookup_table,
        feature_default_value=[-1.0, -1.0],
        max_length=1,
        fail_on_missing_modified_amino_acid=True,
    )

    with pytest.raises(MissingModifiedAminoAcidError):
        feature_extractor._extract_feature(["R[UNIMOD:1][UNIMOD:2]"])


def test_lookup_feature_extractor_fail_on_missing_modified_amino_acid_allows_known_and_unmodified():
    # positive case: with fail_on_missing_modified_amino_acid=True, a modified amino acid
    # present in the lookup table is looked up normally, and plain amino acids /
    # unmodified termini never raise since they are not required to be in the table
    lookup_table = {"M[UNIMOD:35]": [9.0, 9.0]}
    feature_extractor = LookupFeatureExtractor(
        sequence_column_name="sequence",
        feature_column_name="feature",
        lookup_table=lookup_table,
        feature_default_value=[-1.0, -1.0],
        max_length=5,
        fail_on_missing_modified_amino_acid=True,
    )

    sequence_for_lookup = ["[]-", "A", "M[UNIMOD:35]", "-[]"]
    feature = feature_extractor._extract_feature(sequence_for_lookup)

    expected = np.array(
        [
            [-1.0, -1.0],
            [-1.0, -1.0],
            [9.0, 9.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
        ]
    )
    assert np.array_equal(
        feature, expected
    ), "Known modified amino acids and unmodified tokens should resolve without raising."


def test_lookup_feature_extractor_warn_on_missing_modified_amino_acid():
    # warn (not raise) on a missing modified amino acid, still falling back to default
    lookup_table = {"M[UNIMOD:35]": [9.0, 9.0]}
    feature_extractor = LookupFeatureExtractor(
        sequence_column_name="sequence",
        feature_column_name="feature",
        lookup_table=lookup_table,
        feature_default_value=[-1.0, -1.0],
        max_length=4,
        fail_on_missing_modified_amino_acid=False,
    )

    sequence_for_lookup = ["A", "C[UNIMOD:4]"]

    with pytest.warns(MissingModifiedAminoAcidWarning):
        feature = feature_extractor._extract_feature(sequence_for_lookup)

    assert np.array_equal(
        feature[:2], np.array([[-1.0, -1.0], [-1.0, -1.0]])
    ), "A missing modified amino acid should still fall back to the default value."


def test_lookup_feature_extractor_apply_to_split_relaxes_on_eval_split():
    # a strict extractor must still raise on a fit split (train/val), but must only
    # warn (not raise) on a non-fit split (test), so held-out data isn't blocked by a
    # single incomplete lookup-table entry
    lookup_table = {"M[UNIMOD:35]": [9.0, 9.0]}
    feature_extractor = LookupFeatureExtractor(
        sequence_column_name="_parsed_sequence",
        feature_column_name="feature",
        lookup_table=lookup_table,
        feature_default_value=[-1.0, -1.0],
        max_length=4,
        batched=True,
        fail_on_missing_modified_amino_acid=True,
    )

    dataset = Dataset.from_dict(
        {
            "_parsed_sequence": [["A", "C[UNIMOD:4]"]],
            "_n_term_mods": ["[]-"],
            "_c_term_mods": ["-[]"],
        }
    )
    ctx = PipelineContext(
        alphabet={}, num_proc=None, batch_size=10, fit_splits=("train", "val")
    )

    with pytest.raises(MissingModifiedAminoAcidError):
        feature_extractor.apply_to_split(dataset, "train", ctx)

    with pytest.warns(MissingModifiedAminoAcidWarning):
        result = feature_extractor.apply_to_split(dataset, "test", ctx)

    assert np.array_equal(
        result[0]["feature"][:4],
        [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]],
    ), "The test split should fall back to the default value instead of raising an exception."


def test_lookup_feature_extractor_applies_feature_value_offset():
    # feature_value_offset must shift looked-up, defaulted, AND padded values alike
    lookup_table = {"M[UNIMOD:35]": [4.0, 4.0]}
    feature_extractor = LookupFeatureExtractor(
        sequence_column_name="sequence",
        feature_column_name="feature",
        lookup_table=lookup_table,
        feature_default_value=[0.0, 0.0],
        max_length=3,
        feature_value_offset=1,
    )

    feature = feature_extractor._extract_feature(["A", "M[UNIMOD:35]"])

    expected = np.array([[1.0, 1.0], [5.0, 5.0], [1.0, 1.0]])
    assert np.array_equal(feature, expected), (
        "feature_value_offset should shift the looked-up value, the defaulted "
        "unmodified-amino-acid value, and the padded tail value alike."
    )


def test_ptm_loss_gain_lookup_tables_match_and_are_shifted():
    # PTM_LOSS_LOOKUP / PTM_GAIN_LOOKUP are split from the same combined
    # saved_loss_gain_atoms.json file and must cover identical tokens
    assert set(PTM_LOSS_LOOKUP) == set(PTM_GAIN_LOOKUP)

    # spot-check a couple of tokens against their pre-refactor (unshifted) values
    assert PTM_LOSS_LOOKUP["C[UNIMOD:4]"] == [1, 0, 0, 0, 0, 1]
    assert PTM_GAIN_LOOKUP["C[UNIMOD:4]"] == [4, 2, 1, 1, 0, 1]


def test_mod_loss_gain_atom_count_extractors_preserve_original_values():
    # end-to-end regression: even though the lookup tables and feature_default_value
    # were re-baselined to 0, the tensors reaching a downstream model must be
    # numerically identical to before, so existing trained checkpoints keep working
    expected_original_values = {
        "mod_loss": {
            "C[UNIMOD:4]": [2, 1, 1, 1, 1, 2],
            "M[UNIMOD:35]": [4, 2, 1, 1, 1, 2],
        },
        "mod_gain": {
            "C[UNIMOD:4]": [5, 3, 2, 2, 1, 2],
            "M[UNIMOD:35]": [4, 2, 1, 2, 1, 2],
        },
        "atom_count": {
            "C[UNIMOD:4]": [4, 3, 2, 2, 1, 1],
            "M[UNIMOD:35]": [1, 1, 1, 2, 1, 1],
        },
    }

    for feature_name, token_values in expected_original_values.items():
        feature_extractor = LookupFeatureExtractor(
            sequence_column_name="_parsed_sequence",
            feature_column_name=feature_name,
            max_length=1,
            **FEATURE_EXTRACTORS_PARAMETERS[feature_name],
        )
        for token, expected in token_values.items():
            feature = feature_extractor._extract_feature([token])
            assert feature[0].tolist() == [
                float(v) for v in expected
            ], f"{feature_name}[{token}] should be unchanged from before the re-baseline"

        # unmodified amino acids also default to the original (1-shifted) baseline
        default_feature = feature_extractor._extract_feature(["A"])
        assert default_feature[0].tolist() == [1.0] * 6
