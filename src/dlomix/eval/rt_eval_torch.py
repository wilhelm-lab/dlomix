import torch


class TimeDeltaMetric:
    """Class to calculate the time delta metric."""

    def __init__(self, percentage=0.95, normalize=False):
        """
        Parameters
        ----------
        percentage : float, optional
            percentage of absolute error to consider, by default 0.95
        normalize : bool, optional
            whether to normalize the error by the range of y_true, by default False
        """
        self.percentage = percentage
        self.normalize = normalize

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return timedelta(y_true, y_pred, self.percentage, self.normalize)

    def __repr__(self) -> str:
        return (
            f"TimeDeltaMetric(percentage={self.percentage}, normalize={self.normalize})"
        )


def timedelta(
    y_true: torch.Tensor, y_pred: torch.Tensor, percentage=0.95, normalize=False
) -> torch.Tensor:
    """Find error value that is below 95th percentile of absolute error.

    Parameters
    ----------
    y_true : torch.Tensor
        ground truth
    y_pred : torch.Tensor
        predictions
    percentage : float, optional
        percentage of absolute error to consider, by default 0.95
    normalize : bool, optional
        whether to normalize the error by the range of y_true, by default False

    Returns
    -------
    torch.Tensor
        Nth percentile of absolute error, optionally normalized by range of y_true.
    """
    # Note: Flatten before computing abs error.
    # For (batch, 1) tensors, this undercounts total elements, and torch.sort operates along the last dim
    # by default — so on a 2D tensor it sorts within rows, not across all values.
    abs_error = torch.abs(y_true.reshape(-1) - y_pred.reshape(-1))

    # Use .numel() to count total elements after flattening.
    mark_percentile = int(abs_error.numel() * percentage)

    delta = torch.sort(abs_error)[0][mark_percentile - 1]

    if normalize:
        # NOTE: normalize over the flattened y_true for consistency with abs_error.
        norm_range = y_true.reshape(-1).max() - y_true.reshape(-1).min()
        return delta / norm_range

    return delta
