import numpy as np
import tensorflow as tf
import torch

from dlomix.eval.chargestate import (
    adjusted_mean_absolute_error,
    adjusted_mean_squared_error,
)
from dlomix.eval.rt_eval import TimeDeltaMetric, timedelta

Y_TRUE = [0, 1, 2, 2, 0, 0, 0, 0]
Y_PRED = [0, 3, 0, 4, 0, 0, 2, 0]


def test_adjusted_mean_absolute_error():
    y_true = tf.constant(Y_TRUE, dtype="float32")
    y_pred = tf.constant(Y_PRED, dtype="float32")
    mae = adjusted_mean_absolute_error(y_true, y_pred)
    assert np.isclose(mae, 2.0)


def test_adjusted_mean_squared_error():
    y_true = tf.constant(Y_TRUE, dtype="float32")
    y_pred = tf.constant(Y_PRED, dtype="float32")
    mse = adjusted_mean_squared_error(y_true, y_pred)
    assert np.isclose(mse, 4.0)


def test_adjusted_mean_absolute_error_torch():
    y_true = torch.tensor(Y_TRUE, dtype=torch.float32)
    y_pred = torch.tensor(Y_PRED, dtype=torch.float32)
    mae = adjusted_mean_absolute_error(y_true, y_pred)
    assert np.isclose(mae, 2.0)


def test_adjusted_mean_squared_error_torch():
    y_true = torch.tensor(Y_TRUE, dtype=torch.float32)
    y_pred = torch.tensor(Y_PRED, dtype=torch.float32)
    mse = adjusted_mean_squared_error(y_true, y_pred)
    assert np.isclose(mse, 4.0)


def test_rt_eval_rf():
    y_true = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = tf.constant([1.5, 3.0, 4.5, 6.0, 7.5])
    # abs_error =        [0.5, 1.0, 1.5, 2.0, 2.5]

    metric = TimeDeltaMetric(double_delta=True)
    metric.update_state(y_true, y_pred)
    assert metric.delta == 4.0
    assert metric.result() == 4.0 / 1
    assert timedelta(y_true, y_pred) == 2


def test_rt_eval_torch():
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = torch.tensor([1.5, 3.0, 4.5, 6.0, 7.5])
    # abs_error =         [0.5, 1.0, 1.5, 2.0, 2.5]

    assert timedelta(y_true, y_pred) == 2
