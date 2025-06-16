import numpy as np
import torch
import torch.nn.functional as F


def masked_spectral_distance(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the masked spectral distance between true and predicted intensity vectors.
    The masked spectral distance is a metric for comparing the similarity between two intensity vectors.

    Masked, normalized spectral angles between true and pred vectors

    > arccos(1*1 + 0*0) = 0 -> SL = 0 -> high correlation

    > arccos(0*1 + 1*0) = pi/2 -> SL = 1 -> low correlation

    Parameters
    ----------
    y_true : torch.Tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_pred : torch.Tensor
        A tensor containing the predicted values, with the same shape as `y_true`.

    Returns
    -------
    torch.Tensor
        A tensor containing the masked spectral distance between `y_true` and `y_pred`.

    """

    # To avoid numerical instability during training on GPUs,
    # we add a fuzzing constant epsilon of 1×10−7 to all vectors
    epsilon = 1e-7

    # Masking: we multiply values by (true + 1) because then the peaks that cannot
    # be there (and have value of -1 as explained above) won't be considered
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    # L2 norm
    # along last axis / dimension of the tensor
    true_norm = F.normalize(true_masked, p=2, dim=-1)
    pred_norm = F.normalize(pred_masked, p=2, dim=-1)

    # Spectral Angle (SA) calculation
    # (from the definition below, it is clear that ions with higher intensities
    #  will always have a higher contribution)
    product = (pred_norm * true_norm).sum(dim=-1)
    product = torch.clamp(product, -1.0 + epsilon, 1.0 - epsilon)
    arccos = torch.arccos(product)
    batch_losses = 2 * arccos / np.pi

    return batch_losses.mean()


def masked_pearson_correlation_distance(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the masked Pearson correlation distance between true and predicted intensity vectors.
    The masked Pearson correlation distance is a metric for comparing the similarity between two intensity vectors,
    taking into account only the non-negative values in the true values tensor (which represent valid peaks).

    Parameters
    ----------
    y_true : torch.Tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_pred : torch.Tensor
        A tensor containing the predicted values, with the same shape as `y_true`.

    Returns
    -------
    torch.Tensor
        A tensor containing the masked Pearson correlation distance between `y_true` and `y_pred`.

    """

    epsilon = 1e-7

    # Masking: we multiply values by (true + 1) because then the peaks that cannot
    # be there (and have value of -1 as explained above) won't be considered
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    mx = true_masked.mean()
    my = pred_masked.mean()
    xm, ym = true_masked - mx, pred_masked - my
    r_num = (xm * ym).mean()
    r_den = xm.std(unbiased=False) * ym.std(unbiased=False)

    return 1 - (r_num / r_den)
