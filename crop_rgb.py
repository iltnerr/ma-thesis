import cv2
import os


size = 224

dir_string = "data/input/val/"
directory = os.fsencode(dir_string)

for file in os.listdir(directory):
    filename = os.fsdecode(file)

    img = cv2.imread('data/input/val/' + filename)

    H = img.shape[0]
    W = img.shape[1]

    h1 = int(H / 2 - size / 2)
    h2 = int(H / 2 + size / 2)
    w1 = int(W / 2 - size / 2)
    w2 = int(W / 2 + size / 2)

    im_crop = img[h1:h2, w1:w2]

    cv2.imwrite("ddd/" + filename, im_crop)