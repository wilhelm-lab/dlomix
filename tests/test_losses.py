import logging

import tensorflow as tf

from dlomix.losses.intensity import (
    masked_pearson_correlation_distance,
    masked_spectral_distance,
)

logger = logging.getLogger(__name__)


# ------------------ intensity - masked spectral distance ------------------


def test_spectral_distance_identical():
    y_true = [[0.1, 0.2, 0.3]]
    y_true_tensor = tf.convert_to_tensor(y_true)
    y_pred_tensor = tf.convert_to_tensor(y_true)

    sa = masked_spectral_distance(y_true_tensor, y_pred_tensor)
    logger.info("Spectral Angle for identical vectors: {}".format(sa.numpy()))

    assert sa.numpy() == 0


def test_spectral_distance_different():
    y_true = [[0.1, 0.2, 0.3]]
    y_true_tensor = tf.convert_to_tensor(y_true)
    y_pred_tensor = tf.convert_to_tensor([list(reversed(y_true[0]))])

    sa = masked_spectral_distance(y_true_tensor, y_pred_tensor)
    logger.info("Spectral Angle for reversed vectors: {}".format(sa.numpy()))

    assert sa.numpy() != 0


def test_spectral_distance_zero_input():
    y_true = [[0.0, 0.0, 0.0]]
    y_true_tensor = tf.convert_to_tensor(y_true)
    y_pred_tensor = tf.convert_to_tensor(y_true)

    sa = masked_spectral_distance(y_true_tensor, y_pred_tensor)
    logger.info("Spectral Angle for zero input vectors: {}".format(sa.numpy()))


# ------------------ intensity - masked pearson correlation distance ------------------


def test_pearson_correlation_distance_identical():
    y_true = [[0.1, 0.2, 0.3]]
    y_true_tensor = tf.convert_to_tensor(y_true)
    y_pred_tensor = tf.convert_to_tensor(y_true)

    pc = masked_pearson_correlation_distance(y_true_tensor, y_pred_tensor)
    logger.info(
        "Masked Pearson Correlation Distance for identical vectors: {}".format(
            pc.numpy()
        )
    )

    assert pc.numpy() == 0


def test_pearson_correlation_distance_different():
    y_true = [[0.1, 0.2, 0.3]]
    y_true_tensor = tf.convert_to_tensor(y_true)
    y_pred_tensor = tf.convert_to_tensor([list(reversed(y_true[0]))])

    pc = masked_pearson_correlation_distance(y_true_tensor, y_pred_tensor)
    logger.info(
        "Masked Pearson Correlation Distance for reversed vectors: {}".format(
            pc.numpy()
        )
    )

    assert pc.numpy() != 0


def test_pearson_correlation_distance_zero_input():
    y_true = [[0.0, 0.0, 0.0]]
    y_true_tensor = tf.convert_to_tensor(y_true)
    y_pred_tensor = tf.convert_to_tensor(y_true)

    pc = masked_pearson_correlation_distance(y_true_tensor, y_pred_tensor)
    logger.info(
        "Masked Pearson Correlation Distance for zero input vectors: {}".format(
            pc.numpy()
        )
    )


# ---------------------------------------------------------------------------
