import os
import cv2

dir_string = "../data/input/train/"
directory = os.fsencode(dir_string)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    im = cv2.imread(dir_string + filename, 0)
    edges = cv2.Canny(im, 50, 150, apertureSize=3)

    cv2.imwrite("../data/canny/train/" + filename, edges)

"""    cv2.imshow("d", edges)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
"""