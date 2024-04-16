import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import time
import datetime
import platform
import importlib

from settings.config import Home, GETLab
from utils.tools import print_config, avg_kpts_per_image

from preproc.tf_dataset import make_datasets

from evaluation.loss import WCE_from_logits, FocalLoss
from evaluation.metrics import AUC, IOU, Precision, Recall
from evaluation.evaluation import plot_predictions

from utils.tf_utils import get_callbacks
from utils.visualization import plot_kpt_map, plot_histories

from utils.tools import clear_working_dirs, log_results

op_sys = platform.system()
CONFIG = Home if op_sys == "Windows" else GETLab

def train(sess_dict, epochs=1, lr_scheduler=False, lr_decay=False, save_best_model=False, early_stopping=False, save_history=False,
          save_last_model=False, export_predictions=False):

    if export_predictions and not save_best_model:
        raise ValueError("Activate Checkpoints to export predictions!")

    sess_results = {}
    for sess, model in sess_dict.items():

        callbacks = get_callbacks(sess,
                                  early_stopping=early_stopping,
                                  save_best_model=save_best_model,
                                  lr_scheduler=lr_scheduler)

        lr = 5e-5
        if lr_decay:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                lr,
                decay_steps=200,
                decay_rate=0.95,
                staircase=True)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=WCE_from_logits(pos_weight=CONFIG["POS_WEIGHT"]),
                      metrics=[AUC(name="AUC"),
                               IOU(num_classes=2, name="IOU"),
                               Precision(name="Precision"),
                               Recall(name="Recall")
                               ])

        print_config(CONFIG)
        t_start = datetime.datetime.now()
        print(pd.to_datetime(t_start).round('1s'), f"- Start training for session '{sess}'.")

        model.build(input_shape=(CONFIG["BATCH_SIZE"], CONFIG["H"], CONFIG["W"], CONFIG["C"]))

        history = model.fit(ds_train,
                            validation_data=ds_val,
                            epochs=epochs,
                            callbacks=callbacks)

        # export
        if save_history:
            pd.DataFrame.from_dict(history.history).to_csv(f'models/histories/{sess}-history.csv', index=False)
            print(pd.to_datetime(datetime.datetime.now()).round('1s'), f"- Saved history for session '{sess}'.")

        if save_last_model:
            model.save_weights(f"models/checkpoints/{sess}-last_epoch")
            print(pd.to_datetime(datetime.datetime.now()).round('1s'), f"- Saved model weights for session '{sess}'.")

        t_end = datetime.datetime.now()
        t_train = t_end - t_start

        print(pd.to_datetime(t_end).round('1s'),
              f"- Finished Training after ~ {int(t_train.total_seconds() / 60)} minutes for session '{sess}'.")

        if export_predictions:
            plot_predictions(model, f"models/checkpoints/{sess}-best", ds_val, ds_train, sess)

        model.load_weights(f"models/checkpoints/{sess}-best")
        val_metrics = model.evaluate(ds_val, verbose=2, return_dict=True)
        sess_results[sess] = val_metrics

    return sess_results


# DATASET --------------------------------------------------------------------------------------------------------------
ds_train, ds_val = make_datasets(augment_train=CONFIG["AUGMENT_TRAIN_DS"],
                                 crop_images=CONFIG["CROP"])
avg_num_kpts = avg_kpts_per_image(ds_train)/CONFIG["BATCH_SIZE"]
print("avg_num_kpts = ", avg_num_kpts)

"""
for x, y in ds_train:
    plot_kpt_map(x, y)
"""

# TRAINING SESSIONS ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    clear_working_dirs()

    sess_dict = importlib.import_module('settings.sess_' + CONFIG["SESSION_NAME"]).Session

    starttime = datetime.datetime.now()

    results = train(sess_dict,
                    epochs=CONFIG["EPOCHS"],
                    lr_decay=CONFIG["LR_DECAY"],
                    save_best_model=CONFIG["SAVE_BEST_MODEL"],
                    early_stopping=CONFIG["EARLY_STOPPING"],
                    save_history=CONFIG["SAVE_HISTORY"],
                    save_last_model=CONFIG["SAVE_LAST_MODEL"],
                    export_predictions=CONFIG["EXPORT_PREDICTIONS"]
                    )

    traintime = datetime.datetime.now() - starttime

    log_results(results, CONFIG)

    if CONFIG["SAVE_HISTORY"]:
        plot_histories()

    print_config(CONFIG)
    print("avg_num_kpts = ", avg_num_kpts)
    print(f"Training procedure took ~ {int(traintime.total_seconds() / 60)} minutes for all sessions.")