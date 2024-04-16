from models.model import TinyKeyNet, KeyNet, DeepKeyNet, SuperPoint, Unet, UnetMod
from models.architectures import *

Session = {
        "tinykeynet": TinyKeyNet(concat_handcrafted=True),
        "keynet": KeyNet(concat_handcrafted=True),
        "deepkeynet": DeepKeyNet(concat_handcrafted=True),
        "superpoint": SuperPoint(),
        "unet": Unet(**unet_params1),
        "unetmod-pre": UnetMod(use_pretrained_encoder=True, **unetmod_params1),
}

print("settings/baseline.py has been executed")