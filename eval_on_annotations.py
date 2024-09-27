"""
1. read 


0a09e1fd-9436-4724-a93f-e998c55eae82-19C_GRDNSQ SHADE SAIL 3_2024_01_13_12PM_02_37_03.json
0a09e1fd-9436-4724-a93f-e998c55eae82-19C_GRDNSQ SHADE SAIL 3_2024_01_13_12PM_02_37_03.jpg

"""
import csv
import json
import os
from tqdm import tqdm

import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')


DIR = "/home/leeping/Clients/PMY/crowd_counting_test_dataset/"
WEIGHTS_PATH = "weights/SHTechA.pth"


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set parameters for P2PNet evaluation', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--threshold', default=0.5, type=float,
                        help="confidence threshold")
    parser.add_argument('--input_dir', required=True,
                        help='path of directory with images and annotations')
    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default=WEIGHTS_PATH,
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int,
                        help='the gpu used for evaluation')

    return parser


def load_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, device, transform


def inference(img_path, model, device, transform):
    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    # round the size
    width, height = img_raw.size

    if width > height:
        new_width, new_height = 1280, 768
    else:
        new_width, new_height = 768, 1280
    img_raw = img_raw.resize((new_width, new_height), Image.Resampling.LANCZOS)
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    # run inference
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(
        outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    # filter the predictions
    points = outputs_points[outputs_scores >
                            args.threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > args.threshold).sum())

    outputs_scores = torch.nn.functional.softmax(
        outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]
    # draw the predictions
    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(
            img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image

    cv2.imwrite(os.path.join(args.output_dir, os.path.basename(img_path)[:-4] +
                '_pred{}.jpg'.format(predict_cnt)), img_to_draw)

    return len(points)


def main(args):

    model, device, transform = load_model()
    img_list = os.listdir(os.path.join(DIR, "images"))
    results = [
        ["Image", "Ground Truth", "Prediction", "Absolute Error"]
    ]

    for img_filename in tqdm(img_list):
        img_filename_no_ext = img_filename.split(".")[0]
        img_path = os.path.join(DIR, "images", img_filename)
        json_path = os.path.join(
            DIR, "annotation", img_filename_no_ext + ".json")
        with open(json_path, "r") as file:
            annotations = json.load(file)

        pred_count = inference(img_path, model, device, transform)

        results.append([img_filename, annotations["human_num"],
                       pred_count, abs(annotations["human_num"] - pred_count)])

    with open("eval_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        'P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
