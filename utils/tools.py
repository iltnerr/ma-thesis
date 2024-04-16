import tensorflow as tf
import time
import pandas as pd
import numpy as np
import cv2
import os
import glob
import platform

from settings.config import Home, GETLab

op_sys = platform.system()
CONFIG = Home if op_sys == "Windows" else GETLab

def print_config(config):
    print("--------------------------------------[Config]--------------------------------------")
    for k, v in config.items():
        print(k, "=", v)
    print("------------------------------------------------------------------------------------")

def df_to_map(df, H, W):
    """
    Use keypoint DF to generate a score map used for supervised training.
    """
    score_map = np.zeros((H, W))
    for index, row in df.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        s = int(row['s'])
        score_map[y, x] = s

    return score_map

def kpt_path_from_input(input_path):
    """
    Construct path for the corresponding keypoint .csv file, given an input image path.
    """
    op_sys = platform.system()

    file = input_path.split("\\")[-1] if op_sys == "Windows" else input_path.split("/")[-1]
    file = file[:-4]
    dir = input_path.split("\\")[-2] if op_sys == "Windows" else input_path.split("/")[-2]

    # find corresponding .csv in dir
    for root, dirs, files in os.walk("data/keypoints/" + dir):
        kpt_file_l = [f for f in files if f.startswith(file)]

    if CONFIG["IM_TYPE"] == "edges":
        fname = kpt_file_l[0][:-4]
        return "data/keypoints/" + dir + "/" + fname

    # ensure that only a single file is found
    if len(kpt_file_l) < 1:
        raise ValueError("Couldn't find corresponding keypoint .csv file")

    for f in kpt_file_l:
        id = f.split("_")[0]
        if id == file:
            kpt_file = f[:-4]

    return "data/keypoints/" + dir + "/" + kpt_file

def get_image_bw(file, dir, is_input_img):
    im = get_image(file, dir, is_input_img)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im

def get_image(file, dir, is_input_img):
    # in case filename corresponds to an annotated image, rename to original name of input image
    if "_" in file and is_input_img:
        file = file.split("_")[0]

    file = dir + file + ".png"
    img = cv2.imread(file)  # BGR
    return img

def get_kpts(path):
    kp_df = pd.read_csv(path + ".csv", header=None)
    kp_df.columns = ['x', 'y', 's']
    return kp_df

def draw_kpts(image, kpts, filename, is_input_img, scale_koeff=0.3, scale_threshold=None, save_image=False):
    # in case filename corresponds to an annotation, rename to original name of input image
    if "_" in filename and is_input_img:
        filename = filename.split("_")[0]

    # circle props
    color = (255, 0, 0) # Blue
    thickness = 1

    # filter stable keypoints
    if scale_threshold is not None:
        is_stable = kpts['s'] > scale_threshold
        kpts = kpts[is_stable]
        print("Number of Keypoints: " + str(kpts.shape[0]))

    # loop over keypoints
    for index, row in kpts.iterrows():
        radius = int(row["s"] * scale_koeff)
        # radius = 5 # for comparing reasons

        center_coordinates = (int(row["x"]), int(row["y"]))
        image = cv2.circle(image, center_coordinates, radius, color, thickness)

    if save_image:
        if is_input_img:
            cv2.imwrite("./plots/keypoints/inputs/kpts_" + filename + ".png", image)
        else:
            cv2.imwrite("./plots/keypoints/edges/kpts_" + filename + ".png", image)
    else:
        cv2.imshow(filename + ", scale_threshold: " + str(scale_threshold), image)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()

def avg_kpts_per_image(train):
    """
    Returns the average number of keypoints for the train dataset.
    """
    total = 0
    for _, y in train:
        num_kpts = len(tf.where(y > 0))
        total += num_kpts

    average = total/len(list(train))
    return int(average)

def clear_working_dirs():
    dirs = ["models/checkpoints/*", "models/histories/*", "plots/histories/*", "plots/predictions/*"]
    print("REMOVING FILES IN WORKING DIRECTORIES:\n", dirs)

    for directory in dirs:
        files = glob.glob(directory)
        for f in files:
            os.remove(f)

def log_results(results_dict, config):
    with open(f'logs/{time.strftime("%Y%m%d-%H%M%S")}.log', 'w') as f:

        print("----------------[Config]----------------", file=f)
        for k, v in config.items():
            print(k, "=", v, file=f)
        print("----------------------------------------", file=f)

        for session, metrics in results_dict.items():
            print(f"{session} - {metrics}\n", file=f)