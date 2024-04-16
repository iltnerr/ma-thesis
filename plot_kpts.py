import os
from utils.tools import get_kpts, get_image, draw_kpts

ds_split = "train"
kpts_dir = "data/keypoints/" + ds_split + "/"
edges_dir = "data/edges/" + ds_split + "/"
inputs_dir = "data/input/" + ds_split + "/"
se_edges_dir = "data/se_edges/" + ds_split + "/"
segs_dir = "data/segs/" + ds_split + "/"
canny_dir = "data/canny/" + ds_split + "/"

for root, dirs, files in os.walk("data/edges/train/"):
    # must use filename of edge image
    for file in files:
        file = file[:-4]

        # use specific file
        #file = "189011_1"

        # get data for file
        file_path = kpts_dir + file
        kpts = get_kpts(file_path)
        edges = get_image(file, edges_dir, is_input_img=False)  # BGR
        input = get_image(file, inputs_dir, is_input_img=True)  # BGR
        se_edges = get_image(file, se_edges_dir, is_input_img=True)  # BGR
        segs = get_image(file, segs_dir, is_input_img=True)  # BGR
        canny = get_image(file, canny_dir, is_input_img=True)  # BGR

        # draw keypoints
        thresh = 15
        draw_kpts(input, kpts, file, is_input_img=True, scale_threshold=thresh, save_image=True)
        draw_kpts(canny, kpts, file, is_input_img=True, scale_threshold=thresh, save_image=True)
        draw_kpts(se_edges, kpts, file, is_input_img=True, scale_threshold=thresh, save_image=True)
        draw_kpts(segs, kpts, file, is_input_img=True, scale_threshold=thresh, save_image=True)



