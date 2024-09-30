import os
import numpy as np

import torchvision.transforms as standard_transforms
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from .SHHA import SHHA

# DeNormalize used to get original images


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def loading_data(images_dir, annotations_path):
    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])

    # {image basename: np.array of points, ... }
    img_gt_map = create_img_gt_map(annotations_path)
    img_list = list(img_gt_map.keys())
    train_filenames, val_filenames = train_test_split(
        img_list, test_size=0.2, shuffle=True)

    # create the training dataset
    train_set = SHHA(images_dir, train_filenames, img_gt_map, train=True,
                     transform=transform, patch=True, flip=True)
    # create the validation dataset
    val_set = SHHA(images_dir, val_filenames, img_gt_map,
                   train=False, transform=transform)

    return train_set, val_set


def create_img_gt_map(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    img_gt_map = {}

    # <image id="0" name="USOpen/datasets/USOpenTheHandle/images/05b6e03e-2642-4a41-ad23-29ba77a7a996-The_Handle_2023_08_23-14_28_54_0.jpg" width="1280" height="720">
    #     <points label="person" source="manual" occluded="0" points="766.70,286.60" z_order="0">
    #     </points>

    for image_elem in root.findall("image"):
        image_filename = os.path.basename(image_elem.get("name"))
        annotations = []
        for points_elem in list(image_elem.findall("points")):
            if points_elem.get("label") == "person":
                points = points_elem.get("points").split(",")
                points = [float(point) for point in points]
                annotations.append(points)

        if annotations:
            img_gt_map[image_filename] = np.array(annotations)

    return img_gt_map
