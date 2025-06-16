import torch

EPSILON = torch.finfo(torch.float32).eps


def adjusted_mean_absolute_error(
    y_pred: torch.tensor, y_true: torch.tensor
) -> torch.tensor:
    """
    For two vectors, discard those components that
    are 0 in both vectors and compute the mean
    absolute error for the adjusted vector.
    """
    # Convert y_true and y_pred to float tensors
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)

    # Create a mask for elements that are not both zero
    # for torch you can use numpy logic!
    mask = (y_true != 0) & (y_pred != 0)

    # Compute the mean absolute error
    absolute_errors = (y_true - y_pred).abs() * mask

    count_non_zero = mask.sum()

    # Avoid division by zero by adding a small epsilon to the denominator
    mean_absolute_error = (absolute_errors).sum() / (count_non_zero + EPSILON)

    return mean_absolute_error


def adjusted_mean_squared_error(
    y_pred: torch.tensor, y_true: torch.tensor
) -> torch.tensor:
    """
    For two vectors, discard those components that
    are 0 in both vectors and compute the mean
    absolute error for the adjusted vector.
    """
    # Convert y_true and y_pred to float tensors
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)

    # Create a mask for elements that are not both zero
    # for torch you can use numpy logic!
    mask = (y_true != 0) & (y_pred != 0)

    # Compute the mean absolute error
    squared_errors = torch.square((y_true - y_pred) * mask)

    count_non_zero = mask.sum()

    # Avoid division by zero by adding a small epsilon to the denominator
    mean_squared_error = (squared_errors).sum() / (count_non_zero + EPSILON)

    return mean_squared_error
