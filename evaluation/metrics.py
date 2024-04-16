import tensorflow as tf
import numpy as np
import platform
from utils.tf_utils import logits_to_preds

from utils.visualization import plot_kpt_map

from settings.config import Home, GETLab

op_sys = platform.system()
CONFIG = Home if op_sys == "Windows" else GETLab

@tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
def tp_with_distance_threshold(input):
    def adapt_y_true(y_true_pred, threshold=5):
        """
        Adapt the ground truth so that positive predictions are considered correct if they lie within a certain distance
        of the true positive.
        """

        def generate_mask(positions, shape, r):
            x = np.arange(0, shape[1])
            y = np.arange(0, shape[0])

            mask = np.full(shape, False)
            for cy, cx in positions:
                m = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
                mask = np.logical_or(mask, m)

            # plot mask
            """
            mask = mask.astype(int)
            mask_t = tf.convert_to_tensor(mask)
            plot_kpt_map(tf.expand_dims(tf.expand_dims(mask_t, axis=0), axis=-1),
                         tf.expand_dims(y, axis=0))
            """
            return mask

        y_true, y_pred = y_true_pred

        # unbatch
        y_true_unstack = tf.unstack(tf.squeeze(y_true, axis=-1), num=CONFIG["BATCH_SIZE"])
        y_pred_unstack = tf.unstack(tf.squeeze(y_pred, axis=-1), num=CONFIG["BATCH_SIZE"])
        stack_l = []

        for true, pred in zip(y_true_unstack, y_pred_unstack):
            # indices of positives in y_true
            pos_indices_labels = tf.where(true == 1)  # coords in (y, x)

            # circular masks at locations of positives in y_true
            mask = generate_mask(pos_indices_labels, shape=true.shape, r=threshold)

            # filter relevant positives in y_pred
            filtered_preds = tf.where(mask, pred, 0)

            # use filtered_preds and set the corresponding indices in y_true == 1
            true_adapted = tf.where(filtered_preds == 1, 1, true)

            stack_l.append(tf.expand_dims(true_adapted, axis=-1))

        y_true_adapted = tf.stack(stack_l)
        return y_true_adapted

    y_true = tf.numpy_function(adapt_y_true, [input], tf.float32)

    # somehow TF loses the shape information when using tf.numpy_function...
    y_true.set_shape((CONFIG["BATCH_SIZE"], CONFIG["H"], CONFIG["W"], 1))
    return y_true

"""
The following classes are simple wrappers to monitor metrics while training a model which only outputs logits instead of
actual predictions. Also, a distance threshold is used when computing the true positives before evaluating the metrics.
"""

class AUC(tf.keras.metrics.AUC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = logits_to_preds(y_pred)

        #plot_kpt_map(image_t=tf.zeros((1, 224, 224, 3)), map_t=y_true, predictions=y_pred)

        y_true = tp_with_distance_threshold([y_true, y_pred])
        #plot_kpt_map(image_t=tf.zeros((1, 224, 224, 3)), map_t=y_true, predictions=y_pred)

        super(AUC, self).update_state(y_true, y_pred, sample_weight)

class IOU(tf.keras.metrics.MeanIoU):
    def __init__(self, from_logits=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            y_pred = logits_to_preds(y_pred)
            y_true = tp_with_distance_threshold([y_true, y_pred])
            super(IOU, self).update_state(y_true, y_pred, sample_weight)
        else:
            super(IOU, self).update_state(y_true, y_pred, sample_weight)

class Precision(tf.keras.metrics.Precision):
    def __init__(self, from_logits=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            y_pred = logits_to_preds(y_pred)
            y_true = tp_with_distance_threshold([y_true, y_pred])
            super(Precision, self).update_state(y_true, y_pred, sample_weight)
        else:
            super(Precision, self).update_state(y_true, y_pred, sample_weight)

class Recall(tf.keras.metrics.Recall):
    def __init__(self, from_logits=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            y_pred = logits_to_preds(y_pred)
            y_true = tp_with_distance_threshold([y_true, y_pred])
            super(Recall, self).update_state(y_true, y_pred, sample_weight)
        else:
            super(Recall, self).update_state(y_true, y_pred, sample_weight)

class TruePositives(tf.keras.metrics.TruePositives):
    def __init__(self, from_logits=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            y_pred = logits_to_preds(y_pred)
            y_true = tp_with_distance_threshold([y_true, y_pred])
            super(TruePositives, self).update_state(y_true, y_pred, sample_weight)
        else:
            super(TruePositives, self).update_state(y_true, y_pred, sample_weight)

class FalsePositives(tf.keras.metrics.FalsePositives):
    def __init__(self, from_logits=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            y_pred = logits_to_preds(y_pred)
            y_true = tp_with_distance_threshold([y_true, y_pred])
            super(FalsePositives, self).update_state(y_true, y_pred, sample_weight)
        else:
            super(FalsePositives, self).update_state(y_true, y_pred, sample_weight)

class TrueNegatives(tf.keras.metrics.TrueNegatives):
    def __init__(self, from_logits=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            y_pred = logits_to_preds(y_pred)
            y_true = tp_with_distance_threshold([y_true, y_pred])
            super(TrueNegatives, self).update_state(y_true, y_pred, sample_weight)
        else:
            super(TrueNegatives, self).update_state(y_true, y_pred, sample_weight)

class FalseNegatives(tf.keras.metrics.FalseNegatives):
    def __init__(self, from_logits=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            y_pred = logits_to_preds(y_pred)
            y_true = tp_with_distance_threshold([y_true, y_pred])
            super(FalseNegatives, self).update_state(y_true, y_pred, sample_weight)
        else:
            super(FalseNegatives, self).update_state(y_true, y_pred, sample_weight)