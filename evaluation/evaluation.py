import tensorflow as tf
import os
import pandas as pd

from utils.tf_utils import logits_to_preds
from utils.visualization import plot_kpt_map

def evaluate_metrics(y_true, y_pred):
    """
    Evaluates the specified metrics for a models predictions and returns a dict containing the values.
    """

    metrics = {
        #
        # WRONG METRICS! USE CUSTOM ONES WITH TP-DISTANCE-THRESHOLD
        #
        "tp": tf.keras.metrics.TruePositives(),
        "tn": tf.keras.metrics.TrueNegatives(),
        "fp": tf.keras.metrics.FalsePositives(),
        "fn": tf.keras.metrics.FalseNegatives(),
        "iou": tf.keras.metrics.MeanIoU(num_classes=2),
        "precision": tf.keras.metrics.Precision(),
        "recall": tf.keras.metrics.Recall(),
        "auc": tf.keras.metrics.AUC()
    }

    for key, metric in metrics.items():
        metric.update_state(y_true, y_pred)
        metrics[key] = metric.result().numpy()

    metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    return metrics

def plot_predictions(model, weights, ds_val, ds_train, sess):

    model.load_weights(weights)
    counter = 1

    # preds for validation ds
    for x, y in ds_val.take(5):
        logits = model(x, training=False)
        preds = logits_to_preds(logits)

        x_unstack = tf.unstack(x)
        y_unstack = tf.unstack(y)
        preds_unstack = tf.unstack(preds)

        for idx, (image, kpts, preds) in enumerate(zip(x_unstack, y_unstack, preds_unstack)):
            image = tf.expand_dims(image, 0)
            kpts = tf.expand_dims(kpts, 0)
            preds = tf.expand_dims(preds, 0)

            plot_kpt_map(image, kpts, predictions=preds, filename=f"{sess}_{idx}_{counter}")
            counter += 1

    # preds for train ds
    for x_train, y_train in ds_train.take(1):
        logits = model(x_train, training=False)
        preds = logits_to_preds(logits)

        x_unstack = tf.unstack(x_train)
        y_unstack = tf.unstack(y_train)
        preds_unstack = tf.unstack(preds)

        for idx, (image, kpts, preds) in enumerate(zip(x_unstack, y_unstack, preds_unstack)):
            image = tf.expand_dims(image, 0)
            kpts = tf.expand_dims(kpts, 0)
            preds = tf.expand_dims(preds, 0)

            plot_kpt_map(image, kpts, predictions=preds, filename=f"{sess}_{idx}_train")