import tensorflow as tf
L = tf.keras.losses

def cs_loss(y_true, y_pred, sqrt=True, clip=True):
    if clip:
        y_pred = tf.clip_by_value(y_pred, 0, 1)
    if sqrt:
        cond = tf.where(y_true>0)
        gathered = tf.gather_nd(y_true, cond)
        y_true = tf.tensor_scatter_nd_update(y_true, cond, tf.sqrt(gathered))
        cond = tf.where(y_pred>0)
        gathered = tf.gather_nd(y_pred, cond)
        y_pred = tf.tensor_scatter_nd_update(y_pred, cond, tf.sqrt(gathered))
    loss = L.cosine_similarity(y_true, y_pred, axis=-1)
    loss = tf.reduce_mean(loss)
    return loss  
