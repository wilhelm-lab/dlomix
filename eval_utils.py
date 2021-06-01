import tensorflow as tf
import tensorflow.keras.backend as K


def delta95_metric(y_true, y_pred):
    mark95 = tf.cast(
        tf.cast(tf.shape(y_true)[0], dtype=tf.float32) * 0.95, dtype=tf.int32)
    abs_error = K.abs(y_true - y_pred)
    delta = tf.sort(abs_error)[mark95 - 1]
    norm_range = K.max(y_true) - K.min(y_true)
    return (delta * 2) / (norm_range)

def delta99_metric(y_true, y_pred):
    mark99 = tf.cast(
        tf.cast(tf.shape(y_true)[0], dtype=tf.float64) * 0.99, dtype=tf.int32)
    abs_error = tf.abs(y_true - y_pred)
    delta = tf.sort(abs_error)[mark99 - 1]
    norm_range = K.max(y_true) - K.min(y_true)
    return (delta * 2) / (norm_range)


'''
!! too slow due to iteration on the val data !! --> find a better solution
'''


# class Delta95(K.callbacks.Callback):
#
#     def __init__(self, validation_data):
#         super(Delta95).__init__()
#         self.validation_data = validation_data
#         self._data = []
#
#     def on_epoch_end(self, epoch, logs=None):
#
#         y_preds = []
#         y_true = []
#         for X_val, y_val in self.validation_data:
#             y_preds.extend(np.asarray(self.model.predict(X_val)))
#             y_true.extend(y_val)
#
#         print(y_true)
#         print(y_preds)
#         self._data.append(Delta95.delta_tr95(np.array(y_true), np.array(y_preds)))
#
#         return
#
#     def get_data(self):
#         return self._data
#
# # adopted from https://github.com/horsepurve/DeepRTplus/blob/master/RTdata_emb.py
#     @staticmethod
#     def delta_t95(act, pred):
#         num95 = np.array(np.ceil(len(act) * 0.95)).astype(np.int64)
#         return 2 * np.array(sorted(np.abs(act - pred)))[num95 - 1]
#
#     @staticmethod
#     def delta_tr95(act, pred):
#         return Delta95.delta_t95(act, pred) / (np.max(act) - np.min(act))