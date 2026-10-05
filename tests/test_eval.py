"""Behaviour of the shared evaluation metrics.

``dlomix.eval`` has a single ``keras.ops`` implementation shared by both backends, so
this file is backend-neutral — see ``tests/test_losses.py`` for the reasoning and for
how the reference values double as the cross-backend parity check.
"""

import logging

import keras
import numpy as np
import pytest
from keras import ops

from dlomix.eval.chargestate import (
    adjusted_mean_absolute_error,
    adjusted_mean_squared_error,
)
from dlomix.eval.rt_eval import TimeDeltaMetric, timedelta

logger = logging.getLogger(__name__)

Y_TRUE = [[0.1, 0.2, 0.3, 0.0], [0.0, 0.5, 0.25, 0.25]]
Y_PRED = [[0.3, 0.2, 0.1, 0.0], [0.1, 0.4, 0.30, 0.20]]

# Reference values, identical on both backends to within the tolerance below.
EXPECTED = {
    "adjusted_mean_absolute_error": [0.1000000],
    "adjusted_mean_squared_error": [0.0150000],
    "timedelta": [0.2000000],
}
TOLERANCE = dict(rtol=1e-5, atol=1e-6)


def as_numpy(value):
    """Materialise a backend tensor as numpy, whichever framework produced it."""
    return np.asarray(ops.convert_to_numpy(value)).ravel()


@pytest.mark.parametrize(
    "metric_fn",
    [adjusted_mean_absolute_error, adjusted_mean_squared_error, timedelta],
    ids=lambda f: f.__name__,
)
def test_matches_reference_values(metric_fn):
    """Pinned outputs, asserted identically under either backend."""
    result = as_numpy(metric_fn(Y_TRUE, Y_PRED))
    logger.info("%s on %s: %s", metric_fn.__name__, keras.backend.backend(), result)
    np.testing.assert_allclose(result, EXPECTED[metric_fn.__name__], **TOLERANCE)


# ------------------ chargestate - adjusted errors ------------------

SPARSE_TRUE = [0, 1, 2, 2, 0, 0, 0, 0]
SPARSE_PRED = [0, 3, 0, 4, 0, 0, 2, 0]


def test_adjusted_mean_absolute_error():
    assert np.isclose(
        as_numpy(adjusted_mean_absolute_error(SPARSE_TRUE, SPARSE_PRED)), 2.0
    )


def test_adjusted_mean_squared_error():
    assert np.isclose(
        as_numpy(adjusted_mean_squared_error(SPARSE_TRUE, SPARSE_PRED)), 4.0
    )


def test_adjusted_error_keeps_components_nonzero_in_either_vector():
    """Regression: the two backends used to disagree on the mask.

    The PyTorch copy required *both* entries to be non-zero, which discarded real
    prediction errors and made it report lower values than TensorFlow. Only
    components that are zero in **both** vectors are dropped -- here index 0 -- so
    three of the four remain.
    """
    y_true = [0.0, 0.0, 0.7, 0.3]
    y_pred = [0.0, 0.1, 0.6, 0.3]

    # |0-0.1| + |0.7-0.6| + |0.3-0.3| = 0.2, over 3 retained components
    result = as_numpy(adjusted_mean_absolute_error(y_true, y_pred))
    assert np.isclose(result, 0.2 / 3)


def test_adjusted_error_argument_order_is_y_true_then_y_pred():
    """The unified signature is (y_true, y_pred); PyTorch used to take them reversed.

    MAE/MSE are symmetric, so this pins the *masking* asymmetry instead: swapping the
    arguments must not change which components survive.
    """
    y_true = [0.0, 0.0, 0.7, 0.3]
    y_pred = [0.0, 0.1, 0.6, 0.3]

    forward = as_numpy(adjusted_mean_absolute_error(y_true, y_pred))
    reverse = as_numpy(adjusted_mean_absolute_error(y_pred, y_true))
    np.testing.assert_allclose(forward, reverse, **TOLERANCE)


# ------------------ rt_eval - timedelta ------------------


def test_timedelta_function():
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.5, 3.0, 4.5, 6.0, 7.5]
    # abs_error = [0.5, 1.0, 1.5, 2.0, 2.5]; the 95th percentile index lands on 2.0
    assert np.isclose(as_numpy(timedelta(y_true, y_pred)), 2.0)


def test_timedelta_metric_accumulates_and_resets():
    """``TimeDeltaMetric`` is a keras.metrics.Metric, so it works on both backends."""
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.5, 3.0, 4.5, 6.0, 7.5]

    metric = TimeDeltaMetric(double_delta=True)
    metric.update_state(y_true, y_pred)

    assert np.isclose(as_numpy(metric.delta), 4.0)
    assert np.isclose(as_numpy(metric.result()), 4.0)

    metric.reset_state()
    assert np.isclose(as_numpy(metric.delta), 0.0)


def test_timedelta_metric_is_callable_for_one_shot_use():
    """Direct calls skip the streaming state, replacing the old PyTorch callable."""
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.5, 3.0, 4.5, 6.0, 7.5]

    metric = TimeDeltaMetric()
    assert np.isclose(as_numpy(metric(y_true, y_pred)), 2.0)
    assert np.isclose(as_numpy(metric.delta), 0.0)  # untouched


def test_timedelta_metric_config_round_trip():
    """get_config must carry name/dtype through super(), for from_config to work."""
    metric = TimeDeltaMetric(percentage=0.9, double_delta=True, normalize=True)
    config = metric.get_config()

    assert config["percentage"] == 0.9
    assert config["double_delta"] is True
    assert config["normalize"] is True
    assert "name" in config and "dtype" in config

    restored = TimeDeltaMetric.from_config(config)
    assert restored.get_config() == config
