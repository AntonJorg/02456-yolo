import argparse

import os

import time
import torch
import pickle

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from model.dataloader import HELMETDataLoader
from model.dataloader import class_dict
from model.models import Darknet, load_weights, load_darknet_weights
from utils.utils import *


cfg_path = './cfg/yolov3_36.cfg'
weights_path = './weights/darknet53.conv.74'

def get_darknet(img_size, cfg=cfg_path):
    return Darknet(cfg, img_size)


img_size = 416

model = get_darknet(img_size=img_size)

trained_weights_path = "./trained_models_noskip/416e30.pt"
checkpoint = torch.load(trained_weights_path, map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# Load in weights
load_darknet_weights(model, weights_path)


batch_size = 16
num_classes = 36


dataloader_test = HELMETDataLoader("/work1/fbohy/Helmet/", shuffle=True,
                                   batch_size=batch_size, resize=(img_size, img_size), split="test")


cuda_enable = True
cuda_available = torch.cuda.is_available()
if cuda_enable and cuda_available:
    device = torch.device('cuda:0')

    # Maybe we can use this
    # if torch.cuda.device_count() > 1:
    #     print('Using ', torch.cuda.device_count(), ' GPUs')
    #     model = torch.nn.DataParallel(model)
else:
    device = 'cpu'

model.to(device)

opt = {
    'conf_thres': .5,  # Confidence threshold.
    'nms_thres': .45,   # Non-max supression.
    'iou_thres': 0.5   # IoU
}

print('Compute mAP...')
correct = 0
targets = None
outputs, mAPs, TP, confidence, pred_class, target_class = [], [], [], [], [], []
for batch_i, (imgs, targets, annotations) in enumerate(dataloader_test):
    imgs = imgs.to(device)

    with torch.no_grad():
        output = model(imgs)
        output = non_max_suppression(output, conf_thres=opt['conf_thres'], nms_thres=opt['nms_thres'])

    # Compute average precision for each sample
    for sample_i in range(len(targets)):
        correct = []

        # Get labels for sample where width is not zero (dummies)
        annotations = targets[sample_i]
        # Extract detections
        detections = output[sample_i]

        if detections is None:
            # If there are no detections but there are annotations mask as zero AP
            if annotations.size(0) != 0:
                mAPs.append(0)
            continue

        # Get detections sorted by decreasing confidence scores
        detections = detections[np.argsort(-detections[:, 4])]

        # If no annotations add number of detections as incorrect
        if annotations.size(0) == 0:
            target_cls = []
            #correct.extend([0 for _ in range(len(detections))])
            mAPs.append(0)
            continue
        else:
            target_cls = annotations[:, 0]

            # Extract target boxes as (x1, y1, x2, y2)
            target_boxes = xywh2xyxy(annotations[:, 1:5])
            target_boxes *= img_size

            detected = []
            for *pred_bbox, conf, obj_conf, obj_pred in detections:

                pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                # Compute iou with target boxes
                iou = bbox_iou(pred_bbox, target_boxes)
                # Extract index of largest overlap
                best_i = np.argmax(iou)
                # If overlap exceeds threshold and classification is correct mark as correct
                if iou[best_i] > opt['iou_thres'] and obj_pred == annotations[best_i, 0] and best_i not in detected:
                    correct.append(1)
                    detected.append(best_i)
                else:
                    correct.append(0)

        # Compute Average Precision (AP) per class
        AP = ap_per_class(tp=correct, conf=detections[:, 4], pred_cls=detections[:, 6], target_cls=target_cls)

        # Compute mean AP for this image
        mAP = AP.mean()

        # Append image mAP to list
        mAPs.append(mAP)

        # Print image mAP and running mean mAP
        # print('+ Sample [%d/%d] AP: %.4f (%.4f)' % (len(mAPs), len(dataloader) * batch_size, mAP, np.mean(mAPs)))

print('Mean Average Precision: %.4f' % np.mean(mAPs))
