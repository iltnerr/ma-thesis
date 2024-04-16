import tensorflow as tf
from models.model import UnetMod
from models.architectures import unetmod_params1

l2 = tf.keras.regularizers.L2()

Session = {
        "unetmod1_pre_dl": UnetMod(use_dropout=True, kernel_regularizer=l2, use_pretrained_encoder=True, **unetmod_params1),
}

print("settings/sess_opt.py has been executed")