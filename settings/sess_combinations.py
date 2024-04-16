import tensorflow as tf
from models.model import Unet, UnetMod
from models.architectures import *

l2 = tf.keras.regularizers.L2()

Session = {
        # dropout + l2
        "unet1_dl": Unet(use_dropout=True, kernel_regularizer=l2, **unet_params1),
        "unet2_dl": Unet(use_dropout=True, kernel_regularizer=l2, **unet_params2),

        "unetmod1_pre_dl": UnetMod(use_dropout=True, kernel_regularizer=l2, use_pretrained_encoder=True, **unetmod_params1),
        "unetmod2_pre_dl": UnetMod(use_dropout=True, kernel_regularizer=l2, use_pretrained_encoder=True, **unetmod_params2),

        # dropout + l2 + concat gradients
        "unet1_dlg": Unet(use_gradients=True, concat_handcrafted=True, use_dropout=True, kernel_regularizer=l2, **unet_params1),
        "unet2_dlg": Unet(use_gradients=True, concat_handcrafted=True, use_dropout=True, kernel_regularizer=l2, **unet_params2),
}

print("settings/combinations.py has been executed")