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
num_epochs = 3

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
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for sample_batched in dataloaders[phase]:
                inputs = sample_batched['input'].to(device)
                labels = sample_batched['output'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            if phase == 'val':
                val_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history


if __name__ == "__main__":
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/'
    image_datasets = {x: MyDataset(x + '.csv', os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    #print(dataset_sizes)
    # GPU mode
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    """
    # Get a batch of training data
    sample_batched = next(iter(dataset_loaders['train']))
    # Make a grid from batch
    plt.figure()
    show_landmarks_batch(sample_batched)
    plt.axis('off')
    plt.ioff()
    plt.savefig("test.png")
    """
    # Finetuning the convnet
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_outputs)

    model_ft = model_ft.to(device)

    criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # Train and evaluate
    model_ft, train_loss_hist, val_loss_hist = train_model(model_ft, dataset_loaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                           device, num_epochs)
    # plot loss history
    thist = []
    vhist = []

    thist = [h for h in train_loss_hist]
    vhist = [h for h in val_loss_hist]
    plt.title("Training and Validation Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, num_epochs + 1), thist, label="Training")
    plt.plot(range(1, num_epochs + 1), vhist, label="Validation")
    plt.ylim((0, 0.2))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.savefig("loss_curve.png")
    # plt.show()





