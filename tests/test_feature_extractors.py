import logging

import numpy as np

from dlomix.data.processing.feature_extractors import LookupFeatureExtractor

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
