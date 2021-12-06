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

###

# Load in convolutional darknet.
# TODO: Set a default img_size for convenience.


def get_darknet(img_size, cfg=cfg_path):
    return Darknet(cfg, img_size)


img_size = 416

model = get_darknet(img_size=img_size)

trained_weights_path = "./trained_models/416e3L49.798347.pt"
checkpoint = torch.load(trained_weights_path, map_location='cpu')
model.load_state_dict(checkpoint)
model.train()

# Load in weights
load_darknet_weights(model, weights_path)


batch_size = 3

dataloader_train = HELMETDataLoader("./data/HELMET_DATASET_DUMMY", shuffle=True, batch_size=batch_size,
                                    resize=(img_size, img_size), split="training")  # "/work1/fbohy/Helmet/"
dataloader_val = HELMETDataLoader("./data/HELMET_DATASET_DUMMY", shuffle=True,
                                  batch_size=batch_size, resize=(img_size, img_size), split="validation")
dataloader_test = HELMETDataLoader("./data/HELMET_DATASET_DUMMY", shuffle=True,
                                   batch_size=batch_size, resize=(img_size, img_size), split="test")


CUTOFF = 155
# Transfer learning (train only YOLO layers)
for i, (name, p) in enumerate(model.named_parameters()):
    # print(i, p.shape[0], name)
    # if p.shape[0] != 650:  # not YOLO layer
    #    p.requires_grad = False
    if i < CUTOFF:
        p.requires_grad = False


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()))
# optimizer = torch.optim.SGD(
#     filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3,
#     momentum=.9, weight_decay=5e-4, nesterov=True
# )


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

N_EPOCHS = 1
CHECKPOINT_EVERY = 1

num_classes = 36

error_count = 0
for epoch in range(1, N_EPOCHS + 1):
    since = time.time()

    # Train loop
    model.train()
    ui = -1
    rloss = defaultdict(float)  # running loss
    metrics = torch.zeros(4, num_classes)
    for imgs, targets, annotations in dataloader_train:
        # if sum([len(x) for x in targets]) < 1:  # if no targets continue
        #    continue

        optimizer.zero_grad()  # Zero gradients
        loss = model(imgs.to(device), targets, requestPrecision=True)
        try:
            loss.backward()
        except:
            error_count += 1
        optimizer.step()

        # Compute running epoch-means of tracked metrics
        ui += 1
        metrics += model.losses['metrics']
        for key, val in model.losses.items():
            rloss[key] = (rloss[key] * ui + val) / (ui + 1)

        # Precision
        precision = metrics[0] / (metrics[0] + metrics[1] + 1e-16)
        k = (metrics[0] + metrics[1]) > 0
        if k.sum() > 0:
            mean_precision = precision[k].mean().item()
        else:
            mean_precision = 0

        # Recall
        recall = metrics[0] / (metrics[0] + metrics[2] + 1e-16)
        k = (metrics[0] + metrics[2]) > 0
        if k.sum() > 0:
            mean_recall = recall[k].mean().item()
        else:
            mean_recall = 0
    
    train_metric_dict = {
        'epoch': epoch,
        'loss': rloss['loss'],
        'x': rloss['x'],
        'y': rloss['y'],
        'w': rloss['w'],
        'h': rloss['h'],
        'conf': rloss['conf'],
        'cls': rloss['cls'],
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'nT': model.losses['nT'],
        'TP': model.losses['TP'].item(),
        'FP': model.losses['FP'].item(),
        'FN': model.losses['FN'].item()
    }

    # Save metrics
    with open('./loss/train_metrics.pickle', 'wb') as handle:
        pickle.dump(train_metric_dict, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    time_elapsed = time.time() - since
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    if not (epoch - 1) % CHECKPOINT_EVERY:
        torch.save(model.state_dict(), "./trained_models/416" +
                   "e" + str(epoch+3) + ".pt")
    
    # Validation loop
    model.eval()
    ui = -1
    rloss = defaultdict(float)  # running loss
    metrics = torch.zeros(4, num_classes)
    for imgs, targets, annotations in dataloader_val:

        loss = model(imgs.to(device), targets, requestPrecision=True)

        # Compute running epoch-means of tracked metrics
        ui += 1
        metrics += model.losses['metrics']
        for key, val in model.losses.items():
            rloss[key] = (rloss[key] * ui + val) / (ui + 1)

        # Precision
        precision = metrics[0] / (metrics[0] + metrics[1] + 1e-16)
        k = (metrics[0] + metrics[1]) > 0
        if k.sum() > 0:
            mean_precision = precision[k].mean().item()
        else:
            mean_precision = 0

        # Recall
        recall = metrics[0] / (metrics[0] + metrics[2] + 1e-16)
        k = (metrics[0] + metrics[2]) > 0
        if k.sum() > 0:
            mean_recall = recall[k].mean().item()
        else:
            mean_recall = 0
    
    val_metric_dict = {
        'epoch': epoch,
        'loss': rloss['loss'],
        'x': rloss['x'],
        'y': rloss['y'],
        'w': rloss['w'],
        'h': rloss['h'],
        'conf': rloss['conf'],
        'cls': rloss['cls'],
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'nT': model.losses['nT'],
        'TP': model.losses['TP'].item(),
        'FP': model.losses['FP'].item(),
        'FN': model.losses['FN'].item()
    }

    # Save metrics
    with open('./loss/val_metrics.pickle', 'wb') as handle:
        pickle.dump(val_metric_dict, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

# test_loss_stats = []
# for imgs, targets, annotations in dataloader_test:

#     targets = list(map(lambda x: x.type(torch.FloatTensor), targets)) # To correct type.
#     model.eval()
#     loss_dict = model(imgs.to(device), targets, requestPrecision=True).losses
#     test_loss_stats.append(loss_dict)


# with open('loss_stats.pickle', 'wb') as handle:
#     pickle.dump(test_loss_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
