import tensorflow as tf
import tensorflow.keras.backend as K

from dlomix.eval import tf as tf_eval
from dlomix.eval import torch as torch_eval
from dlomix.types import Tensor


# code adopted and modified based on:
# https://github.com/horsepurve/DeepRTplus/blob/cde829ef4bd8b38a216d668cf79757c07133b34b/RTdata_emb.py
def delta95_metric(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Find value that is below 95th percentile of absolute error.
    Scale this by the range of the true values (max-min)

    Parameters
    ----------
    y_true : Tensor
        ground truth
    y_pred : Tensor
        predictions

    Returns
    -------
    Tensor
        95% percentile of absolute error squared divided by range of y_true
    """
    if isinstance(y_true, tf.Tensor):
        ret = tf_eval.rt_eval.delta95_metric(y_true, y_pred)
    else:
        ret = torch_eval.rt_eval.delta95_metric(y_true, y_pred)
    return ret


if __name__ == "__main__":
    # test case: absolute error is 2.0 is  below 95th percentile
    y_true = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = tf.constant([1.5, 3.0, 4.5, 6.0, 7.5])
    # abs_error =        [0.5, 1.0, 1.5, 2.0, 2.5]

    print(delta95_metric(y_true, y_pred))  # 4 / 4
