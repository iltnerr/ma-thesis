# U-Net
unet_params1 = {
    "use_skips": True,
    "architecture": ((32, 3), (64, 3), (128, 3), (256, 3), (512, 3)),
    "use_bias": False,
}

unet_params2 = {
    "use_skips": True,
    "architecture": ((16, 3), (32, 3), (64, 3), (128, 3), (256, 3)),
    "use_bias": False,
}

unet_params3 = {
    "use_skips": True,
    "architecture": ((8, 3), (16, 3), (32, 3), (64, 3), (128, 3)),
    "use_bias": False,
}

unet_params4 = {
    "use_skips": True,
    "architecture": ((4, 3), (8, 3), (16, 3), (32, 3), (64, 3)),
    "use_bias": False,
}

# U-Net Modified
unetmod_params1 = {
    "up_blocks": ((512, 3), (256, 3), (128, 3), (64, 3)),
    "use_bias": False,
}

unetmod_params2 = {
    "up_blocks": ((256, 3), (128, 3), (64, 3), (32, 3)),
    "use_bias": False,
}

unetmod_params3 = {
    "up_blocks": ((128, 3), (64, 3), (32, 3), (16, 3)),
    "use_bias": False,
}

unetmod_params4 = {
    "up_blocks": ((64, 3), (32, 3), (16, 3), (8, 3)),
    "use_bias": False,
}