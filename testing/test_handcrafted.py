import tensorflow as tf
import numpy as np
from models.model import Handcrafted
from utils.tools import get_image_bw, inputs_dir
import cv2
from PIL import Image


def read_bw_image(path):
    img = read_color_image(path)
    img = to_black_and_white(img)
    return img

def to_black_and_white(img):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return im.reshape(im.shape[0], im.shape[1], 1)

def read_color_image(path):
    im_c = cv2.imread(path)
    return im_c.reshape(im_c.shape[0], im_c.shape[1], 3)

# image loading
inputs_dir = inputs_dir + "train/"
image = get_image_bw("41004_3", inputs_dir, is_input_img=True)
#img = Image.fromarray(image)
#img.show()
image = image.astype(float) / image.max()
image = image.reshape(1, image.shape[0], image.shape[1], 1)
image_t = tf.convert_to_tensor(image, dtype=tf.float32)

# keynet loading
"""
im = read_bw_image("../data/input/train/41004.png")
im = im.astype(float) / im.max()
im_scaled = im.reshape(im.shape[0], im.shape[1])
im = im.reshape(1, im.shape[0], im.shape[1], 1)
im_t = tf.convert_to_tensor(im, dtype=tf.float32)
"""

# compute features
handcrafted = Handcrafted()
x = handcrafted(image_t)
x = x.numpy()

# load keynet computations
keynet_feat_A = np.load("../keynet_feat_A.npy")
keynet_feat_B = np.load("../keynet_feat_B.npy")

# comparison
print(np.array_equal(x, keynet_feat_A))
print(np.array_equal(x, keynet_feat_B))
print("----")
atol = 0.0003
rtol = 0
print(np.allclose(x,keynet_feat_A, equal_nan=False, rtol=rtol, atol=atol)) # test if same shape, elements have close enough values
print(np.allclose(x,keynet_feat_B, equal_nan=False, rtol=rtol, atol=atol)) # test if same shape, elements have close enough values


"""
# show feature maps
fmaps = np.split(x, 10, axis=3)
for map in fmaps:
    map = map.reshape(map.shape[1], map.shape[2])
    img = Image.fromarray(map)
    #img.show()
"""

print("ENDE")
