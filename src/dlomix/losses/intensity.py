import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def masked_spectral_distance(y_true, y_pred):
    """Masked, normalized spectral angles between true and pred vectors
    > arccos(1*1 + 0*0) = 0         > SL = 0    > high correlation
    > arccos(0*1 + 1*0) = pi/2      > SL = 1    > low correlation
    """

    # To avoid numerical instability during training on GPUs,
    # we add a fuzzing constant epsilon of 1×10−7 to all vectors
    epsilon = K.epsilon()

    # Masking: we multiply values by (true + 1) because then the peaks that cannot
    # be there (and have value of -1 as explained above) won't be considered
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    # L2 norm
    pred_norm = K.l2_normalize(true_masked, axis=-1)
    true_norm = K.l2_normalize(pred_masked, axis=-1)

    # Spectral Angle (SA) calculation
    # (from the definition below, it is clear that ions with higher intensities
    #  will always have a higher contribution)
    product = K.sum(pred_norm * true_norm, axis=1)
    arccos = tf.math.acos(product)
    return 2 * arccos / np.pi


def masked_pearson_correlation_distance(y_true, y_pred):
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
