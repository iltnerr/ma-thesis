import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

class CMSE(tf.keras.losses.Loss):
    """
    Mean Squared Error with optional weighting of Keypoints.
    """
    def __init__(self, kp_weights=1., name="custom_mse"):
        super().__init__(name=name)
        self.kp_weights = kp_weights

    def call(self, y_true, y_pred):
        if self.kp_weights > 1:
            # assign weights to keypoints if provided
            weights_t = tf.where(y_true > 0, self.kp_weights, 1.)
            mse = tf.math.reduce_mean(tf.math.multiply(tf.square(y_true - y_pred), weights_t))
        else:
            mse = tf.math.reduce_mean(tf.square(y_true - y_pred))

        return mse

class WCE_from_logits(tf.keras.losses.Loss):
    """
    Weighted Cross Entropy. Used for classification tasks where classes are highly imbalanced.
    'pos_weight' allows one to trade off recall and precision by up- or down-weighting the cost of a positive error
    relative to a negative error.
    """
    def __init__(self, pos_weight=1, name="weighted_cross_entropy"):
        super().__init__(name=name)
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        wce = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight))
        return wce

class FocalLoss(tf.keras.losses.Loss):
    """
    See https://arxiv.org/pdf/1708.02002.pdf for more information.

    Focal loss is extremely useful for classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples.
    The loss value is much high for a sample which is misclassified by the classifier as compared to the loss value
    corresponding to a well-classified example. One of the best use-cases of focal loss is its usage in object detection
    where the imbalance between the background class and other classes is extremely high.
    """
    def __init__(self, from_logits=True, alpha=0.25, gamma=2.0, name="Focal_Loss"):
        super().__init__(name=name)
        self.from_logits = from_logits
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=self.from_logits, alpha=self.alpha, gamma=self.gamma)(y_true, y_pred)
        return loss