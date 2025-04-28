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
    Scale this by the range of the true values (max-min)

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
        Percentage percentile of absolute error normalized by range of y_true, if normalize is True
    """
    mark_percentile = int(y_true.shape[0] * percentage)
    abs_error = torch.abs(y_true - y_pred)
    delta = torch.sort(abs_error)[0][mark_percentile - 1]

    if normalize:
        norm_range = torch.max(y_true) - torch.min(y_true)
        return delta / norm_range

    return delta
