import tensorflow as tf


@tf.keras.utils.register_keras_serializable("dlomix")
def adjusted_mean_absolute_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Used as an evaluation metric for charge state prediction.

    For two vectors, discard those components that
    are 0 in both vectors and compute the mean
    absolute error for the adjusted vector.
    """
    # Convert y_true and y_pred to float tensors
    y_true = tf.cast(y_true, dtype="float32")
    y_pred = tf.cast(y_pred, dtype="float32")

    # Create a mask for elements that are not both zero
    mask = tf.cast(tf.not_equal(y_true + y_pred, 0.0), dtype="float32")

    # Apply mask to both y_true and y_pred
    y_true_adjusted = y_true * mask
    y_pred_adjusted = y_pred * mask

    # Compute the mean absolute error
    absolute_errors = tf.abs(y_true_adjusted - y_pred_adjusted)
    sum_absolute_errors = tf.reduce_sum(absolute_errors)
    count_non_zero = tf.reduce_sum(mask)

    # Avoid division by zero by adding a small epsilon to the denominator
    epsilon = tf.keras.backend.epsilon()
    mean_absolute_error = sum_absolute_errors / (count_non_zero + epsilon)

    return mean_absolute_error


@tf.keras.utils.register_keras_serializable("dlomix")
def adjusted_mean_squared_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    For two vectors, discard those components that
    are 0 in both vectors and compute the mean
    squared error for the adjusted vector.
    """
    # Convert y_true and y_pred to float tensors
    y_true = tf.cast(y_true, dtype="float32")
    y_pred = tf.cast(y_pred, dtype="float32")

    # Create a mask for elements that are not both zero
    mask = tf.cast(tf.not_equal(y_true + y_pred, 0.0), dtype="float32")

    # Apply mask to both y_true and y_pred
    y_true_adjusted = y_true * mask
    y_pred_adjusted = y_pred * mask

    # Compute the mean squared error
    squared_errors = tf.square(y_true_adjusted - y_pred_adjusted)
    sum_squared_errors = tf.reduce_sum(squared_errors)
    count_non_zero = tf.reduce_sum(mask)

    # Avoid division by zero by adding a small epsilon to the denominator
    epsilon = tf.keras.backend.epsilon()
    mean_squared_error = sum_squared_errors / (count_non_zero + epsilon)

    return mean_squared_error


if __name__ == "__main__":
    import numpy as np

    y_true = tf.constant([0, 1, 2, 2, 0, 0, 0, 0], dtype="float32")
    y_pred = tf.constant([0, 3, 0, 4, 0, 0, 2, 0], dtype="float32")
    mae = adjusted_mean_absolute_error(y_true, y_pred)
    mse = adjusted_mean_squared_error(y_true, y_pred)
    assert np.isclose(mae, 2.0)
    assert np.isclose(mse, 4.0)
    print(f"Adjusted MAE: {mae:.4f}")
    print(f"Adjusted MSE: {mse:.4f}")
