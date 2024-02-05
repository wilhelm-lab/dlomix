import logging

import numpy as np
import tensorflow as tf

from dlomix.losses import masked_spectral_distance

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


# ---------------------------------------------------------------------------
