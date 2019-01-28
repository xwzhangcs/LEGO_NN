from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class MyTestDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frames.iloc[idx, 0])
        print(img_name)
        #image = io.imread(img_name)
        image = Image.open(img_name)
        # parameters = self.data_frames.iloc[idx, 1:].as_matrix()
        # parameters = parameters.astype('float').reshape(1, -1)
        # parameters = parameters.astype('float')
        if self.transform:
            image = self.transform(image)
            # parameters = torch.from_numpy(parameters)
        sample = {'input': image}
        return sample


