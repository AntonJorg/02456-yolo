import argparse

import os

import time
import torch
import pickle

import matplotlib.pyplot as plt
import numpy as np

from model.dataloader import HELMETDataLoader
from model.dataloader import class_dict
from model.models import Darknet, load_weights, load_darknet_weights
from utils.utils import *

#from IPython.display import clear_output


cfg_path = './cfg/yolov3.cfg'
weights_path = './weights/darknet53.conv.74'




### 

# Load in convolutional darknet.
# TODO: Set a default img_size for convenience.
def get_darknet(img_size, cfg=cfg_path):
    return Darknet(cfg, img_size)

img_size = 832

model = get_darknet(img_size=img_size)

# Load in weights
load_darknet_weights(model, weights_path)


batch_size = 3

dataloader = HELMETDataLoader("./data/HELMET_DATASET_DUMMY", shuffle=True, batch_size=batch_size)#"/work1/fbohy/Helmet/"


CUTOFF = 155
# Transfer learning (train only YOLO layers)
for i, (name, p) in enumerate(model.named_parameters()):
    # print(i, p.shape[0], name)
    # if p.shape[0] != 650:  # not YOLO layer
    #    p.requires_grad = False
    if i > CUTOFF:
        p.requires_grad = False


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
# optimizer = torch.optim.SGD(
#     filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3,
#     momentum=.9, weight_decay=5e-4, nesterov=True
# )


cuda_enable = True
cuda_available = torch.cuda.is_available()
if cuda_enable and cuda_available:
    device = torch.device('cuda:0')
else:
    device = 'cpu'


N_EPOCHS = 1
PLOT_EVERY = 1

model.to(device)

plot_dict = {'train_loss': [], 'train_acc': [], 'Epoch': []}

# imgs, targets, annotations = next(iter(dataloader))
# targets = list(map(lambda x: x.type(torch.FloatTensor), targets)) # To correct type.

for epoch in range(1, N_EPOCHS + 1):
    since = time.time()
    
    for imgs, targets, annotations in dataloader:
    
        targets = list(map(lambda x: x.type(torch.FloatTensor), targets)) # To correct type.

        
        # Train
        model.train()
        optimizer.zero_grad() # Zero gradients
        loss = model(imgs.to(device), targets, requestPrecision=True)
        try:
            loss.backward()
        except:
            with open('errorImgs.pickle', 'wb') as handle:
                pickle.dump(imgs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        optimizer.step()
    time_elapsed = time.time() - since  
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    if not (epoch - 1) % PLOT_EVERY:
        torch.save(model.state_dict(), "./trained_models/testTrained" + str(epoch) + ".pt")
        plot_dict['Epoch'].append(epoch)
        plot_dict['train_loss'].append(loss.cpu().detach().numpy())
        # plot_dict['train_acc'].append(1-trn_err)
        #fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        #ax.plot(plot_dict['Epoch'], plot_dict['train_loss'])
        #ax.set_xlabel('Epochs')
        #ax.set_ylabel('Loss')
        # axs[1].plot(plot_dict['Epoch'], plot_dict['train_acc'])
        # axs[1].set_xlabel('Epochs')
        # axs[1].set_ylabel('Accuracy')
        #plt.show()





with open('loss.pickle', 'wb') as handle:
    pickle.dump(plot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)












