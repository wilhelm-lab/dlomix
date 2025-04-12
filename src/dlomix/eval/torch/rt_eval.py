import torch


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


if __name__ == "__main__":
    # test case: absolute error is 2.0 is  below 95th percentile
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = torch.tensor([1.5, 3.0, 4.5, 6.0, 7.5])
    # abs_error =        [0.5, 1.0, 1.5, 2.0, 2.5]

    print(timedelta(y_true, y_pred))  # 4 / 4
