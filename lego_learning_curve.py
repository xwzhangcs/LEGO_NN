from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
plt.switch_backend('agg')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from lego_processing_data import MyDataset
from PIL import Image
# Ignore warnings
import math
import warnings
warnings.filterwarnings("ignore")

num_outputs = 4
num_epochs = 60
PATH = "lego.pth"

if __name__ == "__main__":
    # plot loss history
    train_loss_hist = [0.0117, 0.0026, 0.0018, 0.0015, 0.0015, 0.0014, 0.0012, 0.0010, 0.0009, 0.0008, 0.0009, 0.0008,
                       0.0007, 0.0007, 0.0006, 0.0006, 0.0005, 0.0005, 0.0005, 0.0005, 0.0002, 0.0002, 0.0002, 0.0002,
                       0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002,
                       0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002,
                       0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002
                       ]
    val_loss_hist = [0.0084, 0.0008, 0.0006, 0.0015, 0.0007, 0.0027, 0.0009, 0.0015, 0.0007, 0.0005, 0.0005, 0.0004,
                     0.0003, 0.0005, 0.0003, 0.0004, 0.0003, 0.0002, 0.0002, 0.0004, 0.0001, 0.0001, 0.0001, 0.0001,
                     0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                     0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                     0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001
                     ]
    thist = []
    vhist = []

    thist = [math.log10(h) for h in train_loss_hist]
    vhist = [math.log10(h) for h in val_loss_hist]
    plt.title("Training and Validation Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, num_epochs + 1), thist, label="Training")
    plt.plot(range(1, num_epochs + 1), vhist, label="Validation")
    # plt.ylim((0, 10.0))
    # plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.savefig("learning_curve.png")
    # plt.show()







