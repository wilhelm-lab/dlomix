"""Behaviour of the shared losses.

``dlomix.losses`` has a single ``keras.ops`` implementation that runs on whichever
backend ``DLOMIX_BACKEND`` selects, so this file is backend-neutral: it feeds plain
Python lists and reads results back with ``keras.ops.convert_to_numpy``.

Cross-backend agreement is checked by the reference values in ``EXPECTED``. CI runs
this file under both backends (the ``build`` job on TensorFlow, the ``torch-backend``
job on PyTorch); the two runs asserting the same numbers *is* the parity check.
"""

import logging

import keras
import numpy as np
import pytest
from keras import ops

from dlomix.losses.intensity import (
    masked_pearson_correlation_distance,
    masked_spectral_distance,
)
from dlomix.losses.ionmob import MaskedIonmobLoss

logger = logging.getLogger(__name__)

Y_TRUE = [[0.1, 0.2, 0.3, 0.0], [0.0, 0.5, 0.25, 0.25]]
Y_PRED = [[0.3, 0.2, 0.1, 0.0], [0.1, 0.4, 0.30, 0.20]]

# Reference values, identical on both backends to within the tolerance below.
EXPECTED = {
    "masked_spectral_distance": [0.4935034, 0.1590250],
    "masked_pearson_correlation_distance": [0.3135935],
}
TOLERANCE = dict(rtol=1e-5, atol=1e-6)


def as_numpy(value):
    """Materialise a backend tensor as numpy, whichever framework produced it."""
    return np.asarray(ops.convert_to_numpy(value)).ravel()


@pytest.mark.parametrize(
    "loss_fn",
    [masked_spectral_distance, masked_pearson_correlation_distance],
    ids=lambda f: f.__name__,
)
def test_matches_reference_values(loss_fn):
    """Pinned outputs, asserted identically under either backend."""
    result = as_numpy(loss_fn(Y_TRUE, Y_PRED))
    logger.info("%s on %s: %s", loss_fn.__name__, keras.backend.backend(), result)
    np.testing.assert_allclose(result, EXPECTED[loss_fn.__name__], **TOLERANCE)


def test_accepts_numpy_and_native_tensors():
    """The shared losses take plain arrays as well as the active backend's tensors."""
    from_lists = as_numpy(masked_spectral_distance(Y_TRUE, Y_PRED))
    from_numpy = as_numpy(
        masked_spectral_distance(
            np.array(Y_TRUE, dtype="float32"), np.array(Y_PRED, dtype="float32")
        )
    )
    from_native = as_numpy(
        masked_spectral_distance(
            ops.convert_to_tensor(Y_TRUE), ops.convert_to_tensor(Y_PRED)
        )
    )

    np.testing.assert_allclose(from_lists, from_numpy, **TOLERANCE)
    np.testing.assert_allclose(from_lists, from_native, **TOLERANCE)


# ------------------ intensity - masked spectral distance ------------------


def test_spectral_distance_is_per_sample():
    """One value per sample, the Keras convention (PyTorch used to pre-reduce)."""
    assert as_numpy(masked_spectral_distance(Y_TRUE, Y_PRED)).shape == (2,)


def test_spectral_distance_identical():
    sa = as_numpy(masked_spectral_distance([[0.1, 0.2, 0.3]], [[0.1, 0.2, 0.3]]))
    logger.info("Spectral Angle for identical vectors: %s", sa)
    assert sa[0] == 0


def test_spectral_distance_different():
    sa = as_numpy(masked_spectral_distance([[0.1, 0.2, 0.3]], [[0.3, 0.2, 0.1]]))
    logger.info("Spectral Angle for reversed vectors: %s", sa)
    assert sa[0] != 0


def test_spectral_distance_zero_input():
    sa = as_numpy(masked_spectral_distance([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]))
    logger.info("Spectral Angle for zero input vectors: %s", sa)
    assert np.isfinite(sa).all()


def test_spectral_distance_perfect_match_is_finite_and_near_zero():
    """Clipping before arccos rules out NaN on an exact match.

    It cannot force an exact 0 for every input: arccos has an infinite derivative at
    1, so a normalized dot product landing on 0.99999994 in float32 still yields
    ~3e-4. That is inherent to the metric, not to the unification.
    """
    y = np.random.default_rng(3).random((8, 12)).astype("float32")

    result = as_numpy(masked_spectral_distance(y, y))

    assert np.isfinite(result).all()
    assert (result >= 0).all()
    np.testing.assert_allclose(result, 0.0, atol=1e-3)


# ------------------ intensity - masked pearson correlation distance ------------------


def test_pearson_correlation_distance_identical():
    pc = as_numpy(
        masked_pearson_correlation_distance([[0.1, 0.2, 0.3]], [[0.1, 0.2, 0.3]])
    )
    logger.info("Masked Pearson Correlation Distance, identical vectors: %s", pc)
    assert pc[0] == 0


def test_pearson_correlation_distance_different():
    pc = as_numpy(
        masked_pearson_correlation_distance([[0.1, 0.2, 0.3]], [[0.3, 0.2, 0.1]])
    )
    logger.info("Masked Pearson Correlation Distance, reversed vectors: %s", pc)
    assert pc[0] != 0


def test_pearson_correlation_distance_zero_input():
    pc = as_numpy(
        masked_pearson_correlation_distance([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    )
    logger.info("Masked Pearson Correlation Distance, zero input vectors: %s", pc)


# ------------------ differentiability ------------------


def test_gradients_flow_for_the_active_backend():
    """The shared loss is differentiable on whichever backend Keras is using.

    ``keras.ops`` builds graph nodes for the active backend only, so the gradient
    machinery has to match it -- torch autograd when KERAS_BACKEND is torch, a
    GradientTape otherwise.
    """
    rng = np.random.default_rng(11)
    y_true = rng.random((4, 6)).astype("float32")
    y_pred = rng.random((4, 6)).astype("float32")

    if keras.backend.backend() == "torch":
        import torch

        predictions = torch.tensor(y_pred, requires_grad=True)
        masked_spectral_distance(torch.tensor(y_true), predictions).mean().backward()
        gradient = predictions.grad
        assert gradient is not None and torch.isfinite(gradient).all()
    else:
        import tensorflow as tf

        predictions = tf.Variable(y_pred)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                masked_spectral_distance(tf.constant(y_true), predictions)
            )
        gradient = tape.gradient(loss, predictions)
        assert gradient is not None
        assert bool(tf.reduce_all(tf.math.is_finite(gradient)))


# MaskedIonmobLoss: CCS error over all samples plus CCS-std error over the samples
# whose std target is not -1 (MSE: 0.25, 0, 1 -> 0.41667; 0.25, 1 -> 0.625).
IONMOB_OUTPUTS = ([[1.0], [2.0], [3.0]], [[0.5], [1.0], [2.0]])
IONMOB_TARGETS = ([[1.5], [2.0], [2.0]], [[1.0], [-1.0], [1.0]])


@pytest.mark.parametrize(
    "use_mse, std_targets, expected",
    [
        (True, IONMOB_TARGETS[1], 0.41666667 + 0.625),
        (False, IONMOB_TARGETS[1], 0.5 + 0.75),
        (True, [[-1.0], [-1.0], [-1.0]], 0.41666667),  # no std target: std term is 0
    ],
)
def test_masked_ionmob_loss_reference_values(use_mse, std_targets, expected):
    outputs = tuple(ops.convert_to_tensor(np.float32(o)) for o in IONMOB_OUTPUTS)
    loss = MaskedIonmobLoss(use_mse=use_mse)(outputs, (IONMOB_TARGETS[0], std_targets))
    np.testing.assert_allclose(as_numpy(loss), [expected], **TOLERANCE)


def test_masked_ionmob_loss_accepts_flat_targets():
    # datasets yield (batch,) targets for the (batch, 1) model outputs
    outputs = tuple(ops.convert_to_tensor(np.float32(o)) for o in IONMOB_OUTPUTS)
    targets = tuple(np.ravel(t) for t in IONMOB_TARGETS)
    loss = MaskedIonmobLoss()(outputs, targets)
    np.testing.assert_allclose(as_numpy(loss), [0.41666667 + 0.625], **TOLERANCE)
