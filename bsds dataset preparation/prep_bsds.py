"""
Convert BSDS .mat files into .png files and prepare the data for keypoint extraction:
- binary images -> white=edge
- add black border
- rotation of 90° to have images with same shape

Save .png files to "data/BSDS500_groundTruth"
"""

import scipy.io
from matplotlib import pyplot as plt
from scipy import ndimage
import os

import cv2

from shutil import copyfile


def get_images(matfile):
    # .mat -> np.array
    mat_contents = scipy.io.loadmat(matfile, struct_as_record=False, squeeze_me=True)
    groundtruth = mat_contents['groundTruth']
    return [gt.Boundaries for gt in groundtruth]

def save_png(img, name, id, dir):
    # add black border and also rotate if necessary
    # img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    img = rot(img)

    # invert image, because edge tracing considers white as edges
    img = 1 - img

    name = name.split(dir + "/", 1)[1]
    name = name[:-4] + "_" + str(id)
    dir = "../data/BSDS500_groundTruth/" + dir
    plt.imsave(dir + "/" + name + '.png', img, cmap="binary")

def rot(image):
    dim = 321
    if image.shape[0] != dim:
        image = ndimage.rotate(image, 90)
    return image

def get_paths():
    mat_paths = {"test": [], "train": [], "val": []}
    for directory in ["test", "train", "val"]:
        for root, dirs, files in os.walk("../data/BSR_bsds500/BSR/BSDS500/data/groundTruth/" + directory):
            mat_paths[directory] = [root + "/" + file for file in files]
    return mat_paths

def check_input_images():
    for directory in ["test", "train", "val"]:
        print("Checking inputs:" + directory + "...")
        for root, dirs, files in os.walk("../data/BSR_bsds500/BSR/BSDS500/data/images/" + directory):
            for file in files:
                # rotate if necessary and save to data/input/
                img = cv2.imread(root + "/" + file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = rot(img)
                dir = "../data/input/" + directory
                file = file[:-4]
                plt.imsave(dir + "/" + file + ".png", img)


def choose_image(matfile):
    """
    Find Image with average segment count from a list of Images.
     - sort list
     - if number of images is even: choose the image with higher segment count
     - if number of images is odd: choose middle image

    Return Index of chosen Image.
    """

    mat_contents = scipy.io.loadmat(matfile, struct_as_record=False, squeeze_me=True)
    groundtruth = mat_contents['groundTruth']
    candidates = [gt.Segmentation for gt in groundtruth]
    segs = [cand.max() for cand in candidates]

    # find middle
    sorted_segs = sorted(segs)
    to_choose = sorted_segs[len(sorted_segs) // 2]

    # return index of the chosen image
    idx = segs.index(to_choose)
    return idx

def copy_image(id, dir, file):
    file = file.split(dir + "/", 1)[1]
    file = file[:-4] + "_" + str(id)
    dst = "../data/middle segs/" + dir + "/" + file + ".png"
    src = "../data/BSDS500_groundTruth/" + dir + "/" + file + ".png"

    copyfile(src, dst)


# rotate inputs to match annotated image dimensions
# check_input_images()

# convert .mat files to .png and save in separate directory
mat_files = get_paths()
"""
for directory, files in mat_files.items():
    print("Converting " + directory + "...")
    for file in files:
        boundaries = get_images(file)
        for idx, img in enumerate(boundaries):
            save_png(img, file, idx, directory)
"""

# loop over .mat files and find Images with average number of Segments. Then copy the selected images
# from "data/BSDS500_groundTruth" to "data/middle segs/"
for directory, files in mat_files.items():
    for file in files:
        idx_to_choose = choose_image(file)
        copy_image(idx_to_choose, directory, file)






