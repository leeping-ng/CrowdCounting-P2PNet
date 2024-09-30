import os
import scipy.io
import numpy as np
from tqdm import tqdm

MODE = "test"  # "train"
DIR = "/home/leeping/Downloads/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/" + MODE + "_data/"
LIST_FILEPATH = "/home/leeping/Repos/crowd-counting-comparison/CrowdCounting-P2PNet/data/" + MODE + ".list"
list_to_write = []


annotations_dir = os.path.join(DIR, "ground_truth")
annotations_list = os.listdir(annotations_dir)
output_dir = os.path.join(DIR, "images")

for annotation_filename in tqdm(annotations_list):
    mat = scipy.io.loadmat(os.path.join(annotations_dir, annotation_filename))
    arr = mat["image_info"][0][0][0][0][0]
    # remove "GT_" in front
    txt_filename = annotation_filename.split(".")[0][3:] + ".txt"
    img_filename = annotation_filename.split(".")[0][3:] + ".jpg"
    txt_filepath = os.path.join(
        DIR, "images", txt_filename)
    np.savetxt(txt_filepath, arr)
    list_to_write.append(MODE + "/" + img_filename +
                         " " + MODE + "/" + txt_filename)


with open(LIST_FILEPATH, "w") as f:
    for line in list_to_write:
        f.write(f"{line}\n")
