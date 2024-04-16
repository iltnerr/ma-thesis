crop = True
wce_weight = 40
session_name = "opt"

Home = {
    "CFG_NAME": "Home",
    "SESSION_NAME": session_name,

    # inputs
    "IM_TYPE": "segs", # "edges", "input", "canny", "se_edges" or "segs"
    "H": 224 if crop else 320,
    "W": 224 if crop else 480,
    "C": 3,
    "CROP": crop,
    "AUGMENT_TRAIN_DS": False,
    "KPT_SCALE_THRESHOLD": 15,

    # training
    "EPOCHS": 2,
    "BATCH_SIZE": 1,
    "EARLY_STOPPING": True,
    "LR_SCHEDULER": False,
    "LR_DECAY": False,

    # export
    "SAVE_HISTORY": False,
    "SAVE_LAST_MODEL": False,
    "SAVE_BEST_MODEL": False,
    "EXPORT_PREDICTIONS": False,

    # hyperparams
    "POS_WEIGHT": wce_weight,
}

GETLab = {
    "CFG_NAME": "GETLab",
    "SESSION_NAME": session_name,

    # inputs
    "IM_TYPE": "segs", # "edges", "input", "canny", "se_edges" or "segs"
    "H": 224 if crop else 320,
    "W": 224 if crop else 480,
    "C": 3,
    "CROP": crop,
    "AUGMENT_TRAIN_DS": False,
    "KPT_SCALE_THRESHOLD": 15,

    # training
    "EPOCHS": 100,
    "BATCH_SIZE": 2,
    "EARLY_STOPPING": True,
    "LR_SCHEDULER": False,
    "LR_DECAY": True,

    # export
    "SAVE_HISTORY": True,
    "SAVE_LAST_MODEL": False,
    "SAVE_BEST_MODEL": True,
    "EXPORT_PREDICTIONS": True,

    # hyperparams
    "POS_WEIGHT": wce_weight,
}