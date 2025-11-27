import functools

import numpy as np
import tensorflow as tf

from ..losses import masked_spectral_distance


def reshape_dims(array):
    n, dims = array.shape
    assert dims == 174
    nlosses = 1
    return array.reshape([array.shape[0], 30 - 1, 2, nlosses, 3])


def reshape_flat(array):
    s = array.shape
    flat_dim = [s[0], functools.reduce(lambda x, y: x * y, s[1:], 1)]
    return array.reshape(flat_dim)


def normalize_base_peak(array):
    # flat
    maxima = array.max(axis=1)
    array = array / maxima[:, np.newaxis]
    return array


def mask_outofrange(array, lengths, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        array[i, lengths[i] - 1 :, :, :, :] = mask
    return array


def mask_outofcharge(array, charges, mask=-1.0):
    # dim
    for i in range(array.shape[0]):
        if charges[i] < 3:
            array[i, :, :, :, charges[i] :] = mask
    return array


def calculate_spectral_angle(true, pred, batch_size=600):
    """

    Calculates masked spectral distance normalized to [0, 1] range.
    Values of -1 in true are treated as masked (missing peaks).

    Args:
        true: Ground truth array of shape (n, num_values) or array of arrays
        pred: Predicted array of shape (n, num_values) or array of arrays
        batch_size: Kept for API compatibility with old tensorflow 1 version (not used)

    Returns:
        Array of shape (n,) with spectral angles in range [0, 1]
    """
    # Handle pandas Series with array values (common case)
    if hasattr(true, "values"):
        true = true.values
    if hasattr(pred, "values"):
        pred = pred.values

    # Convert array of arrays to 2D array if needed
    if true.dtype == object:
        true = np.stack(true)
    if pred.dtype == object:
        pred = np.stack(pred)

    # Ensure proper float dtype
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    n = true.shape[0]
    epsilon = 1e-7  # Matches K.epsilon() default

    # Masking: multiply by (true + 1) to mask out -1 values
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)

    # L2 normalization (note: original has swapped variables, reproducing exactly)
    # Original: pred_norm = K.l2_normalize(true_masked, axis=-1)
    #           true_norm = K.l2_normalize(pred_masked, axis=-1)
    pred_norm = true_masked / (
        np.linalg.norm(true_masked, axis=-1, keepdims=True) + epsilon
    )
    true_norm = pred_masked / (
        np.linalg.norm(pred_masked, axis=-1, keepdims=True) + epsilon
    )

    # Spectral angle calculation
    product = np.sum(pred_norm * true_norm, axis=1)
    product = np.clip(product, -1.0, 1.0)  # Numerical stability
    arccos = np.arccos(product)

    # Normalize to [0, 1] range
    spectral_distance = 2 * arccos / np.pi

    # Original code does: sa = 1 - s.run(sa_graph)
    # So we invert it here to match exactly
    sa = 1 - spectral_distance

    # Handle NaN values
    sa = np.nan_to_num(sa)

    return sa


def get_spectral_angle(true, pred, batch_size=600):
    """Legacy sepctral angle calculation using TensorFlow 1.x session"""
    n = true.shape[0]
    sa = np.zeros([n])

    def iterate():
        if n > batch_size:
            for i in range(n // batch_size):
                true_sample = true[i * batch_size : (i + 1) * batch_size]
                pred_sample = pred[i * batch_size : (i + 1) * batch_size]
                yield i, true_sample, pred_sample
            i = n // batch_size
            yield i, true[(i) * batch_size :], pred[(i) * batch_size :]
        else:
            yield 0, true, pred

    for i, t_b, p_b in iterate():
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as s:
            sa_graph = masked_spectral_distance(t_b, p_b)
            sa_b = 1 - s.run(sa_graph)
            sa[i * batch_size : i * batch_size + sa_b.shape[0]] = sa_b
    sa = np.nan_to_num(sa)
    return sa


def normalize_intensity_predictions(
    data,
    sequence_column_name="sequences",
    labels_column_name="intensities_raw",
    predictions_column_name="intensities_pred",
    precursor_charge_column_name="precursor_charge_onehot",
    batch_size=600,
    compute_spectral_angle=True,
    use_legacy_tf_sa_fn=False,
):
    assert (
        sequence_column_name in data
    ), "Key sequences is missing in the data provided for post-processing"
    assert (
        predictions_column_name in data
    ), "Key intensities_pred is missing in the data provided for post-processing"
    assert (
        precursor_charge_column_name in data
    ), "Key precursor_charge_onehot is missing in the data provided for post-processing"

    sequence_lengths = data[sequence_column_name].apply(lambda x: len(x))
    intensities = np.stack(data[predictions_column_name].to_numpy()).astype(np.float32)
    precursor_charge_onehot = np.stack(data[precursor_charge_column_name].to_numpy())
    charges = list(precursor_charge_onehot.argmax(axis=1) + 1)

    intensities[intensities < 0] = 0
    intensities = reshape_dims(intensities)
    intensities = mask_outofrange(intensities, sequence_lengths)
    intensities = mask_outofcharge(intensities, charges)
    intensities = reshape_flat(intensities)
    m_idx = intensities == -1
    intensities = normalize_base_peak(intensities)
    intensities[m_idx] = -1
    data[predictions_column_name] = intensities.tolist()

    if labels_column_name in data and compute_spectral_angle:
        if use_legacy_tf_sa_fn:
            data["spectral_angle"] = get_spectral_angle(
                np.stack(data[labels_column_name].to_numpy()).astype(np.float32),
                intensities,
                batch_size=batch_size,
            )
        else:
            data["spectral_angle"] = calculate_spectral_angle(
                np.stack(data[labels_column_name].to_numpy()).astype(np.float32),
                intensities,
                batch_size=batch_size,
            )
    return data
