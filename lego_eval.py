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
import warnings
warnings.filterwarnings("ignore")

num_outputs = 4
num_epochs = 10
PATH = "lego.pth"

if __name__ == "__main__":
    # Define transforms
    transformations = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])
    # Define custom dataset
    test_data = MyDataset("test.csv", "data/test", transformations)
    # Define data loader
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=4)
    # GPU mode
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    # load model
    model = torch.load(PATH)

    # test model
    model.eval()
    for i_batch, test_batch in enumerate(test_dataset_loader):
        inputs = test_batch['input'].to(device)
        labels = test_batch['output'].to(device)
        print(i_batch, inputs.size(),
              labels.size())
        outputs = model(inputs)
        print("labels is {}, and outputs is {}".format(labels, outputs))
        loss = criterion(outputs.float(), labels.float())
        print("loss is {}".format(loss))
        if i_batch == 3:
            break







