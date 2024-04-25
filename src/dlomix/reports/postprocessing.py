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


def get_spectral_angle(true, pred, batch_size=600):
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


def normalize_intensity_predictions(data, batch_size=600):
    assert (
        "sequences" in data
    ), "Key sequences is missing in the data provided for post-processing"
    assert (
        "intensities_pred" in data
    ), "Key intensities_pred is missing in the data provided for post-processing"
    assert (
        "precursor_charge_onehot" in data
    ), "Key precursor_charge_onehot is missing in the data provided for post-processing"

    sequence_lengths = data["sequences"].apply(lambda x: len(x))
    intensities = np.stack(data["intensities_pred"].to_numpy()).astype(np.float32)
    precursor_charge_onehot = np.stack(data["precursor_charge_onehot"].to_numpy())
    charges = list(precursor_charge_onehot.argmax(axis=1) + 1)

    intensities[intensities < 0] = 0
    intensities = reshape_dims(intensities)
    intensities = mask_outofrange(intensities, sequence_lengths)
    intensities = mask_outofcharge(intensities, charges)
    intensities = reshape_flat(intensities)
    m_idx = intensities == -1
    intensities = normalize_base_peak(intensities)
    intensities[m_idx] = -1
    data["intensities_pred"] = intensities.tolist()

    if "intensities_raw" in data:
        data["spectral_angle"] = get_spectral_angle(
            np.stack(data["intensities_raw"].to_numpy()).astype(np.float32),
            intensities,
            batch_size=batch_size,
        )
    return data
