import tensorflow as tf
from models.model import Unet, UnetMod
from models.architectures import *

l2 = tf.keras.regularizers.L2()

Session = {
        # dropout
        "unet1_d": Unet(use_dropout=True, **unet_params1),
        "unet2_d": Unet(use_dropout=True, **unet_params2),

        "unetmod1_pre_d": UnetMod(use_pretrained_encoder=True, use_dropout=True, **unetmod_params1),
        "unetmod2_pre_d": UnetMod(use_pretrained_encoder=True, use_dropout=True, **unetmod_params2),

        # l2
        "unet1_l": Unet(kernel_regularizer=l2, **unet_params1),
        "unet2_l": Unet(kernel_regularizer=l2, **unet_params2),

        "unetmod1_pre_l": UnetMod(use_pretrained_encoder=True, kernel_regularizer=l2, **unetmod_params1),
        "unetmod2_pre_l": UnetMod(use_pretrained_encoder=True, kernel_regularizer=l2, **unetmod_params2),
}

print("settings/regularized.py has been executed")