import tensorflow as tf
import numpy as np
import platform

from settings.config import Home, GETLab
from utils.tools import kpt_path_from_input, get_kpts, df_to_map
from utils.visualization import plot_kpt_map

op_sys = platform.system()
CONFIG = Home if op_sys == "Windows" else GETLab

def make_datasets(augment_train=False, crop_images=False):
    """
    Create Train, Test, Val Dataset Splits.
    """
    paths_train = tf.data.Dataset.list_files("data/" + CONFIG["IM_TYPE"] + "/train/*", shuffle=True)
    paths_val = tf.data.Dataset.list_files("data/" + CONFIG["IM_TYPE"] + "/val/*", shuffle=True)

    ds_train = paths_train.map(lambda path: tf.py_function(func=prepare_dataset,
                                                           inp=[path],
                                                           Tout=[tf.float32, tf.float32]),
                                                           num_parallel_calls=tf.data.AUTOTUNE)

    ds_val = paths_val.map(lambda path: tf.py_function(func=prepare_dataset,
                                                       inp=[path],
                                                       Tout=[tf.float32, tf.float32]),
                                                       num_parallel_calls=tf.data.AUTOTUNE)

    # augment training data
    if augment_train:
        ds_train = augment_training_data(ds_train)

    # crop images to square size
    if crop_images:
        ds_train = ds_train.map(lambda x, y: tf.py_function(func=square_center_crop,
                                                            inp=[x, y, CONFIG["H"]], # take height as size for image
                                                            Tout=[tf.float32, tf.float32]),
                                num_parallel_calls=tf.data.AUTOTUNE)

        ds_val = ds_val.map(lambda x, y: tf.py_function(func=square_center_crop,
                                                        inp=[x, y, CONFIG["H"]], # take height as size for image
                                                        Tout=[tf.float32, tf.float32]),
                            num_parallel_calls=tf.data.AUTOTUNE)

    # only use dataset elements with at least 5 keypoints
    ds_train = ds_train.filter(filter_fn)
    ds_val = ds_val.filter(filter_fn)

    # set shapes
    ds_train = ds_train.map(set_shapes)
    ds_val = ds_val.map(set_shapes)

    # batch, prefetch, cache
    ds_train = configure_for_performance(ds_train, CONFIG["BATCH_SIZE"])
    ds_val = configure_for_performance(ds_val, CONFIG["BATCH_SIZE"])

    return ds_train, ds_val

def prepare_dataset(path):
    """
    Wrapper function to prepare the dataset.
    """
    image_t, kpts_path = process_path(path)
    image_t, kpts_map_t = preproc_data(image_t, kpts_path,
                                       scale_threshold=CONFIG["KPT_SCALE_THRESHOLD"])
    return image_t, kpts_map_t

def process_path(tf_string):
    """Use file paths of the tf.data.Dataset to get corresponding images and keypoint paths."""
    file = tf.io.read_file(tf_string)
    image_t = tf.io.decode_image(file, channels=3, dtype=tf.dtypes.float32)  # RGB

    # get corresponding keypoint file path
    file_path = tf_string.numpy().decode('utf-8')
    kpts_path = kpt_path_from_input(file_path)
    return image_t, kpts_path

def preproc_data(image, kpts_path, scale_threshold=0):
    """
    Prepare data for training.
    """
    # image
    im_H = image.shape[0]
    im_W = image.shape[1]

    # keypoints
    kpts_df = get_kpts(kpts_path)
    # filter stable keypoints if a scale threshold > 0 is provided
    if scale_threshold > 0:
        is_stable = kpts_df['s'] > scale_threshold
        kpts_df = kpts_df[is_stable]

    score_map = df_to_map(kpts_df, im_H, im_W)
    score_map_t = tf.convert_to_tensor(score_map, dtype=tf.float32) # shape (im_H, im_W)
    score_map_t = tf.reshape(score_map_t, [score_map_t.shape[0], score_map_t.shape[1], 1]) # shape (im_H, im_W, 1)

    # score_map contains scale values at keypoint locations, binary_map contains 1s at keypoint locations.
    binary_map = tf.where(score_map_t > 0, 1., 0)

    # resize image and keypoint map, because encoder-decoder architectures need inputs to be divisible by powers of 2
    binary_map = binary_map[:320, :480]
    image = image[:320, :480]

    return image, binary_map

def configure_for_performance(ds, batch_size, buffer_size=tf.data.AUTOTUNE):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True) # buffer size !>= dataset size
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(buffer_size)
  return ds

def augment_training_data(ds_train):
    def augment(image_label, seed):
        image, label = image_label

        # Make a new seed
        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

        # plot_kpt_map(image, label, greyscale=False)

        # Random brightness
        image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed)
        image = tf.clip_by_value(image, 0, 1)
        # plot_kpt_map(image, label, greyscale=False)

        # Random hue
        image = tf.image.stateless_random_hue(image, max_delta=0.5, seed=seed)
        # plot_kpt_map(image, label, greyscale=False)

        # Random saturation
        image = tf.image.stateless_random_saturation(image, lower=0.5, upper=1.0, seed=seed)
        # plot_kpt_map(image, label, greyscale=False)

        # Random contrast
        image = tf.image.stateless_random_contrast(image, lower=0.7, upper=0.9, seed=seed)

        # need to expand dims for keras layer
        # image = tf.expand_dims(image, 0)
        # label = tf.expand_dims(label, 0)
        # plot_kpt_map(image, label, greyscale=False)
        """
        image = tf.keras.layers.experimental.preprocessing.RandomRotation(0.4, fill_mode='reflect', seed=seed,
                                                                          interpolation='bilinear')(image)
        label = tf.keras.layers.experimental.preprocessing.RandomRotation(0.4, fill_mode='reflect', seed=seed,
                                                                          interpolation='nearest')(label)
        """
        # plot_kpt_map(image, label)

        # Random flip horizontally
        image = tf.image.stateless_random_flip_left_right(image, seed=new_seed)
        label = tf.image.stateless_random_flip_left_right(label, seed=new_seed)
        # plot_kpt_map(image, label, greyscale=False)

        # Random flip vertically
        image = tf.image.stateless_random_flip_up_down(image, seed=new_seed)
        label = tf.image.stateless_random_flip_up_down(label, seed=new_seed)
        # plot_kpt_map(image, label, greyscale=False)

        return image, label

    counter = tf.data.experimental.Counter()
    ds_augmented = tf.data.Dataset.zip((ds_train, (counter, counter)))
    ds_augmented = ds_augmented.map(augment)
    ds_extended = tf.data.experimental.sample_from_datasets([ds_augmented, ds_train], weights=[0.5, 0.5])
    return ds_extended

def square_center_crop(image, kpt_map, size):
    H = image.shape[0]
    W = image.shape[1]

    h1 = int(H/2-size/2)
    h2 = int(H/2+size/2)
    w1 = int(W/2-size/2)
    w2 = int(W/2+size/2)

    im_crop = image[h1:h2, w1:w2, :]
    kpt_map_crop = kpt_map[h1:h2, w1:w2, :]

    return im_crop, kpt_map_crop

def filter_fn(image, kpt_map):
    """
    Filter function to remove elements with unsufficient keypoints from the dataset.

    DO NOT REMOVE 'image' ARG!
    """
    kpts = tf.where(kpt_map > 0)
    return True if len(kpts) > 5 else False

def set_shapes(image, label):
    """
    Tensorflow loses the shape information when using custom datasets (map functions etc.).
    Therefore, set shapes manually after processing the datasets.
    """
    image.set_shape((CONFIG["H"], CONFIG["W"], CONFIG["C"]))
    label.set_shape((CONFIG["H"], CONFIG["W"], 1))
    return image, label