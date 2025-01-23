from keras import backend as K


def adjusted_mean_absolute_error(y_true, y_pred):
    """
    Used as an evaluation metric for charge state prediction.

    For two vectors, discard those components that
    are 0 in both vectors and compute the mean
    absolute error for the adjusted vector.
    """
    # Convert y_true and y_pred to float tensors
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")

    # Create a mask for elements that are not both zero
    mask = K.cast(K.not_equal(y_true + y_pred, 0.0), dtype="float32")

    # Apply mask to both y_true and y_pred
    y_true_adjusted = y_true * mask
    y_pred_adjusted = y_pred * mask

    # Compute the mean absolute error
    absolute_errors = K.abs(y_true_adjusted - y_pred_adjusted)
    sum_absolute_errors = K.sum(absolute_errors)
    count_non_zero = K.sum(mask)

    # Avoid division by zero by adding a small epsilon to the denominator
    epsilon = K.epsilon()
    mean_absolute_error = sum_absolute_errors / (count_non_zero + epsilon)

    return mean_absolute_error


def adjusted_mean_squared_error(y_true, y_pred):
    """
    For two vectors, discard those components that
    are 0 in both vectors and compute the mean
    squared error for the adjusted vector.
    """
    # Convert y_true and y_pred to float tensors
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")

    # Create a mask for elements that are not both zero
    mask = K.cast(K.not_equal(y_true + y_pred, 0.0), dtype="float32")

    # Apply mask to both y_true and y_pred
    y_true_adjusted = y_true * mask
    y_pred_adjusted = y_pred * mask

    # Compute the mean squared error
    squared_errors = K.square(y_true_adjusted - y_pred_adjusted)
    sum_squared_errors = K.sum(squared_errors)
    count_non_zero = K.sum(mask)

    # Avoid division by zero by adding a small epsilon to the denominator
    epsilon = K.epsilon()
    mean_squared_error = sum_squared_errors / (count_non_zero + epsilon)

    return mean_squared_error
