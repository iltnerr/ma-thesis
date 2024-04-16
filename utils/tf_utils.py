import tensorflow as tf
import platform

from settings.config import Home, GETLab

op_sys = platform.system()
CONFIG = Home if op_sys == "Windows" else GETLab


def get_callbacks(session, early_stopping=False, m_monitor="val_loss", m_mode="min", patience=30,
                  restore_weights=False, save_best_model=False, lr_scheduler=False):
    """
    Create callbacks list for model.fit()
    """

    callbacks_l = []

    if early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=m_monitor,
                                                                   patience=patience,
                                                                   min_delta=0,
                                                                   verbose=1,
                                                                   mode=m_mode,
                                                                   restore_best_weights=restore_weights)
        callbacks_l.append(early_stopping_callback)

    if save_best_model:
        # save best model over the whole training procedure
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"models/checkpoints/{session}-best",
            save_weights_only=True,
            monitor=m_monitor,
            mode=m_mode,
            save_best_only=True)
        callbacks_l.append(model_checkpoint_callback)

    if lr_scheduler:
        scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule)
        callbacks_l.append(scheduler_callback)

    if not callbacks_l:
        callbacks_l = None

    return callbacks_l

def schedule(epoch):
    if epoch < 30:
        return 1e-3
    elif epoch >= 30 and epoch < 60:
        return 1e-4
    else:
        return 1e-5

def NMS(prob, size):
    """Performs non maximum suppression on the heatmap using max-pooling. This method is
    faster than box_nms, but does not suppress contiguous that have the same probability
    value.
    Arguments:
        prob: the probability heatmap, with shape `[N, H, W, C]`.
        size: a scalar, the size of the pooling window.
    """

    with tf.name_scope('spatial_nms'):
        pooled = tf.nn.max_pool(
                prob, ksize=[1, size, size, 1], strides=[1, 1, 1, 1], padding='SAME')
        prob = tf.where(tf.equal(prob, pooled), prob, tf.zeros_like(prob))
        return prob

def logits_to_preds(logits, nms_size=15):
    """
    Logits -> Probabilities -> NMS -> Thresholding (-> Binary Predictions)
    """
    prob = tf.nn.sigmoid(logits)
    pred = NMS(prob, nms_size)
    pred = tf.round(pred)
    return pred