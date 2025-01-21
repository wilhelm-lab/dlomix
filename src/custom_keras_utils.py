from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.losses import categorical_crossentropy, mean_squared_error

def adjusted_mean_absolute_error(y_true, y_pred):
    '''
    For two vectors, discard those components that 
    are 0 in both vectors and compute the mean 
    absolute error for the adjusted vector.
    '''
    # Convert y_true and y_pred to float tensors
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    # Create a mask for elements that are not both zero
    mask = K.cast(K.not_equal(y_true + y_pred, 0.0), dtype='float32')

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
    '''
    For two vectors, discard those components that 
    are 0 in both vectors and compute the mean 
    squared error for the adjusted vector.
    '''
    # Convert y_true and y_pred to float tensors
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    # Create a mask for elements that are not both zero
    mask = K.cast(K.not_equal(y_true + y_pred, 0.0), dtype='float32')

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


def upscaled_mean_squared_error(y_true, y_pred):
    '''
    For two vectors, multiply each element by 100 
    and compute the mean squared error for the 
    upscaled vector.
    '''
    # Convert y_true and y_pred to float tensors
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    upscale_factor = 100
    y_true_upscaled = y_true * upscale_factor
    y_pred_upscaled = y_pred * upscale_factor

    return mean_squared_error(y_true_upscaled, y_pred_upscaled)


def masked_spectral_distance(y_true, y_pred):
    """
    ### Function stolen from dlomix, but adjusted to work with our CSD vectors

    Calculates the masked spectral distance between true and predicted intensity vectors.
    The masked spectral distance is a metric for comparing the similarity between two intensity vectors.

    Masked, normalized spectral angles between true and pred vectors

    > arccos(1*1 + 0*0) = 0 -> SL = 0 -> high correlation

    > arccos(0*1 + 1*0) = pi/2 -> SL = 1 -> low correlation

    Parameters
    ----------
    y_true : tf.Tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_pred : tf.Tensor
        A tensor containing the predicted values, with the same shape as `y_true`.

    Returns
    -------
    tf.Tensor
        A tensor containing the masked spectral distance between `y_true` and `y_pred`.

    """

    # Convert y_true and y_pred to float tensors
    #y_true = y_true[:, 0]
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    # To avoid numerical instability during training on GPUs,
    # we add a fuzzing constant epsilon of 1×10−7 to all vectors
    epsilon = K.epsilon()

    # Masking: we multiply values by (true + 1) because then the peaks that cannot
    # be there (and have value of -1 as explained above) won't be considered
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    # L2 norm
    pred_norm = K.l2_normalize(true_masked + epsilon, axis=-1)
    true_norm = K.l2_normalize(pred_masked + epsilon, axis=-1)

    # Spectral Angle (SA) calculation
    # (from the definition below, it is clear that ions with higher intensities
    #  will always have a higher contribution)
    product = K.sum(pred_norm * true_norm, axis=1)
    product = K.clip(product, -1.0 + epsilon, 1.0 - epsilon)
    arccos = tf.math.acos(product)
    return 2 * arccos / np.pi


def masked_spectral_angle(y_true, y_pred):
    return 1 - masked_spectral_distance(y_true, y_pred)


def masked_pearson_correlation_distance(y_true, y_pred):
    """
    ### Function stolen from dlomix, but adjusted to work with our CSD vectors

    Calculates the masked Pearson correlation distance between true and predicted intensity vectors.
    The masked Pearson correlation distance is a metric for comparing the similarity between two intensity vectors,
    taking into account only the non-negative values in the true values tensor (which represent valid peaks).

    Parameters
    ----------
    y_true : tf.Tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_pred : tf.Tensor
        A tensor containing the predicted values, with the same shape as `y_true`.

    Returns
    -------
    tf.Tensor
        A tensor containing the masked Pearson correlation distance between `y_true` and `y_pred`.

    """

    # Convert y_true and y_pred to float tensors
    #y_true = y_true[:, 0]
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    epsilon = K.epsilon()

    # Masking: we multiply values by (true + 1) because then the peaks that cannot
    # be there (and have value of -1 as explained above) won't be considered
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    mx = tf.math.reduce_mean(true_masked)
    my = tf.math.reduce_mean(pred_masked)
    xm, ym = true_masked - mx, pred_masked - my
    r_num = tf.math.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return 1 - (r_num / r_den)


def masked_pearson_correlation(y_true, y_pred):
    return 1 - masked_pearson_correlation_distance(y_true, y_pred)


def euclidean_distance_loss(y_true, y_pred):
    """
    From https://riptutorial.com/keras/example/32022/euclidean-distance-loss
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def euclidean_similarity(y_true, y_pred):
    return 1 - euclidean_distance_loss(y_true, y_pred) / np.sqrt(2)


def smoothed_csd_mean_absolute_error(y_true, y_pred):
    return adjusted_mean_absolute_error(y_true[:, 0], y_pred) + adjusted_mean_absolute_error(y_true[:, 1], y_pred)


def smoothed_csd_mean_squared_error(y_true, y_pred):
    return adjusted_mean_squared_error(y_true[:, 0], y_pred) + adjusted_mean_squared_error(y_true[:, 1], y_pred)


def smoothed_csd_categorical_crossentropy(y_true, y_pred):
    return categorical_crossentropy(y_true[:, 0], y_pred) + categorical_crossentropy(y_true[:, 1], y_pred)


def smoothed_csd_upscaled_mean_squared_error(y_true, y_pred):
    return upscaled_mean_squared_error(y_true[:, 0], y_pred) + upscaled_mean_squared_error(y_true[:, 1], y_pred)


def smoothed_csd_masked_spectral_distance(y_true, y_pred):
    def internal_f(y_true, y_pred):
        # Convert y_true and y_pred to float tensors
        y_true = K.cast(y_true, dtype='float32')
        y_pred = K.cast(y_pred, dtype='float32')

        # To avoid numerical instability during training on GPUs,
        # we add a fuzzing constant epsilon of 1×10−7 to all vectors
        epsilon = K.epsilon()

        # Masking: we multiply values by (true + 1) because then the peaks that cannot
        # be there (and have value of -1 as explained above) won't be considered
        pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
        true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

        # L2 norm
        pred_norm = K.l2_normalize(true_masked + epsilon, axis=-1)
        true_norm = K.l2_normalize(pred_masked + epsilon, axis=-1)

        # Spectral Angle (SA) calculation
        # (from the definition below, it is clear that ions with higher intensities
        #  will always have a higher contribution)
        product = K.sum(pred_norm * true_norm, axis=1)
        product = K.clip(product, -1.0 + epsilon, 1.0 - epsilon)
        arccos = tf.math.acos(product)
        return 2 * arccos / np.pi
    return internal_f(y_true[:, 0], y_pred) + internal_f(y_true[:, 1], y_pred)
