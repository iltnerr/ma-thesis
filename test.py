import tensorflow as tf
import pandas as pd
import numpy as np

from preproc.tf_dataset import make_datasets, prepare_dataset, square_center_crop, filter_fn, augment_training_data

from models.model import KeyNet, TinyKeyNet, DeepKeyNet, SuperPoint, Unet, UnetMod, Handcrafted
from models.architectures import *

from evaluation.loss import CMSE, WCE_from_logits, FocalLoss
from evaluation.evaluation import evaluate_metrics
from evaluation.metrics import AUC, IOU, Precision, Recall, tp_with_distance_threshold

from settings.config import Home, GETLab

from utils.tf_utils import logits_to_preds
from utils.visualization import plot_kpt_map
from utils.tools import avg_kpts_per_image, log_results

CONFIG = Home
"""
paths_train = tf.data.Dataset.list_files("data/segs/train/*", shuffle=True)
for path in paths_train:
    image_t, kpts_map_t = prepare_dataset(path)
"""

ds_train, ds_val = make_datasets(augment_train=CONFIG["AUGMENT_TRAIN_DS"],
                                 crop_images=CONFIG["CROP"])
avg_num_kpts = avg_kpts_per_image(ds_train)


l2 = tf.keras.regularizers.L2()
model = UnetMod(use_dropout=True, kernel_regularizer=l2, use_pretrained_encoder=True, **unetmod_params1)
model.build(input_shape=(CONFIG["BATCH_SIZE"], CONFIG["H"], CONFIG["W"], CONFIG["C"]))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=WCE_from_logits(pos_weight=CONFIG["POS_WEIGHT"]),
              metrics=[AUC(name="AUC"),
                       IOU(num_classes=2, name="IOU"),
                       Precision(name="Precision"),
                       Recall(name="Recall")
                       ])

model.load_weights("models/weights/unetmod1_pre_dl-best")
#model.evaluate(ds_val, verbose=2)

i=0
for x, y in ds_val:
    logits = model(x)
    preds = logits_to_preds(logits)
    plot_kpt_map(x, y, predictions=preds, filename=str(i))
    i += 1


