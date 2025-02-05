import torch


def delta95_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Find error value that is below 95th percentile of absolute error.
    Scale this by the range of the true values (max-min)

    Parameters
    ----------
    y_true : torch.Tensor
        ground truth
    y_pred : torch.Tensor
        predictions

    Returns
    -------
    torch.Tensor
        absolute error squared divided by range of y_true
    """
    mark95 = int(y_true.shape[0] * 0.95)
    abs_error = torch.abs(y_true - y_pred)
    delta = torch.sort(abs_error)[0][mark95 - 1]
    norm_range = torch.max(y_true) - torch.min(y_true)
    return (delta * 2) / norm_range


if __name__ == "__main__":
    # test case: absolute error is 2.0 is  below 95th percentile
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = torch.tensor([1.5, 3.0, 4.5, 6.0, 7.5])
    # abs_error =        [0.5, 1.0, 1.5, 2.0, 2.5]

    print(delta95_metric(y_true, y_pred))  # 4 / 4
