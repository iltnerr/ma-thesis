import tensorflow as tf
import datetime
import platform
import pickle
from tensorflow import keras
import kerastuner as kt

from settings.config import Home, GETLab

from preproc.tf_dataset import make_datasets

from models.model import Unet, UnetMod
from models.architectures import *

from evaluation.loss import WCE_from_logits, FocalLoss
from evaluation.metrics import AUC, IOU, Precision, Recall

from utils.tf_utils import get_callbacks

op_sys = platform.system()
CONFIG = Home if op_sys == "Windows" else GETLab

def model_builder(hp):
    # search space
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-4, step=1e-5)
    hp_pos_weight = hp.Int('pos_weight', min_value=20, max_value=60, step=5)
    hp_lr_steps = hp.Int('lr_steps', min_value=150, max_value=300, step=25)
    hp_lr_decay = hp.Float('lr_decay', min_value=0.85, max_value=0.95, step=0.01)

    model = UnetMod(use_dropout=True, kernel_regularizer=tf.keras.regularizers.L2(), use_pretrained_encoder=True, **unetmod_params1)

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hp_learning_rate,
        decay_steps=hp_lr_steps,
        decay_rate=hp_lr_decay,
        staircase=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=WCE_from_logits(pos_weight=hp_pos_weight),
                  metrics=[AUC(name="AUC"),
                           IOU(num_classes=2, name="IOU"),
                           Precision(name="Precision"),
                           Recall(name="Recall")
                           ])
    return model

if __name__ == "__main__":

    project = 'random_search'

    ds_train, ds_val = make_datasets(augment_train=CONFIG["AUGMENT_TRAIN_DS"],
                                     crop_images=CONFIG["CROP"])

    tuner = kt.RandomSearch(model_builder,
                            objective="val_loss",
                            max_trials=100,
                            directory='opt',
                            project_name=project
                            )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                  patience=30,
                                                  min_delta=0,
                                                  verbose=1,
                                                  mode="min"
                                                  )

    starttime = datetime.datetime.now()

    tuner.search(ds_train,
                 validation_data=ds_val,
                 epochs=70,
                 callbacks=[stop_early])

    traintime = datetime.datetime.now() - starttime
    print(f"Hyperparameter optimization took ~ {int(traintime.total_seconds() / 60)} minutes.")

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    pickle.dump(best_hps, open(f"opt/{project}/best_hparams.pkl", "wb"))
    print("Best Hyperparameters:\n", best_hps.values)