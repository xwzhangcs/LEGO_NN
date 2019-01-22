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


class ToTensor(object):
    def __call__(self, sample):
        image, parameters = sample['input'], sample['output']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'input': torch.from_numpy(image),
                'output': torch.from_numpy(parameters)}


class MyDataset(Dataset):
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
        image = io.imread(img_name)
        parameters = self.data_frames.iloc[idx, 1:].as_matrix()
        # parameters = parameters.astype('float').reshape(1, -1)
        parameters = parameters.astype('float')
        if self.transform:
            image = self.transform(image)
            parameters = torch.from_numpy(parameters)
        sample = {'input': image, 'output': parameters}
        return sample


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    images_batch, paras_batch = sample_batched['input'], sample_batched['output']
    batch_size = len(images_batch)
    grid = utils.make_grid(images_batch)
    grid = grid.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    grid = std * grid + mean
    grid = np.clip(grid, 0, 1)
    plt.imshow(grid)
    for i in range(batch_size):
        print(paras_batch[i, :, :].numpy())


if __name__ == "__main__":
    # Define transforms
    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])
    # Define custom dataset
    training_data= MyDataset("test.csv", "data/test", transformations)
    # Define data loader
    mn_dataset_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=3, shuffle=False, num_workers=0)

    for i_batch, sample_batched in enumerate(mn_dataset_loader):
        print(i_batch, sample_batched['input'].size(),
              sample_batched['output'].size())


