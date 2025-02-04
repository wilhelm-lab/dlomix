import numpy as np
from keras import backend as K

from dlomix.eval.chargestate import (
    adjusted_mean_absolute_error,
    adjusted_mean_squared_error,
)

Y_TRUE = [0, 1, 2, 2, 0, 0, 0, 0]
Y_PRED = [0, 3, 0, 4, 0, 0, 2, 0]


def test_adjusted_mean_absolute_error():
    y_true = K.constant(Y_TRUE, dtype="float32")
    y_pred = K.constant(Y_PRED, dtype="float32")
    y_true, y_pred = K.to_dense(y_true), K.to_dense(y_pred)
    mae = adjusted_mean_absolute_error(y_true, y_pred)
    assert np.isclose(mae, 2.0)


def test_adjusted_mean_squared_error():
    y_true = K.constant(Y_TRUE, dtype="float32")
    y_pred = K.constant(Y_PRED, dtype="float32")
    y_true, y_pred = K.to_dense(y_true), K.to_dense(y_pred)
    mse = adjusted_mean_squared_error(y_true, y_pred)
    assert np.isclose(mse, 4.0)
