import scipy.io
from matplotlib import pyplot as plt
from scipy import ndimage
import os
import numpy as np
import cv2
from shutil import copyfile

def get_paths():
    mat_paths = {"test": [], "train": [], "val": []}
    for directory in ["test", "train", "val"]:
        for root, dirs, files in os.walk("../data/BSR_bsds500/BSR/BSDS500/data/groundTruth/" + directory):
            mat_paths[directory] = [root + "/" + file for file in files]
    return mat_paths

def gen_lut():
  """
  Generate a label colormap compatible with opencv lookup table, based on
  Rick Szelski algorithm in `Computer Vision: Algorithms and Applications`,
  appendix C2 `Pseudocolor Generation`.
  :Returns:
    color_lut : opencv compatible color lookup table
  """
  tobits = lambda x, o: np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)
  arr = np.arange(256)
  r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
  g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
  b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
  return np.concatenate([[[b]], [[g]], [[r]]]).T

def labels2rgb(labels, lut):
  """
  Convert a label image to an rgb image using a lookup table
  :Parameters:
    labels : an image of type np.uint8 2D array
    lut : a lookup table of shape (256, 3) and type np.uint8
  :Returns:
    colorized_labels : a colorized label image
  """
  return cv2.LUT(cv2.merge((labels, labels, labels)), lut)


mat_paths = get_paths()

for split, files in mat_paths.items():
    for file in files:
        filename = file.split(split + "/")[1]
        mat_contents = scipy.io.loadmat("../data/BSR_bsds500/BSR/BSDS500/data/groundTruth/" + split + "/" + filename,
                                        struct_as_record=False, squeeze_me=True)
        groundtruth = mat_contents['groundTruth']
        candidates = [gt.Segmentation for gt in groundtruth]
        segs = [cand.max() for cand in candidates]

        # find middle
        sorted_segs = sorted(segs)
        to_choose = sorted_segs[len(sorted_segs) // 2]
        idx = segs.index(to_choose)
        im_indexes = candidates[idx]

        labels = np.arange(256).astype(np.uint8)[np.newaxis, :]
        lut = gen_lut()
        rgb = labels2rgb(labels, lut)

        # indexed image to rgb
        image = np.zeros((im_indexes.shape[0], im_indexes.shape[1], 3))  # black RGB image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                color_index = im_indexes[i, j]
                image[i, j] = rgb[0, color_index] # RGB

        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # rotate if necessary
        if image.shape[0] != 321:
            image = ndimage.rotate(image, 90)

        cv2.imwrite("../data/seg/" + split + "/" + filename[:-4] + ".png", image) # BGR