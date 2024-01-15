'''
datasets.py
- Script to load the datasets
'''

import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
import torch

import sys

import utils

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '/scratch/pytorch_data/', dataset_name: str = 'CIFAR10', batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_data_loader_workers = 8

        # CIFAR10 is RGB, MNIST and FashionMNIST are grayscale
        # if self.dataset_name == 'CIFAR10':
        #     self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        #     ])
        #
        # else:
        #     self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         #transforms.Normalize((0.5,), (0.5,))
        #     ])
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform = DequantizeTransform(data_transforms) # Add those transforms to the dequantize transform

    def prepare_data(self):
        # Download datasets
        if self.dataset_name == 'CIFAR10':
            datasets.CIFAR10(self.data_dir, train=True, download=True)
            datasets.CIFAR10(self.data_dir, train=False, download=True)
        elif self.dataset_name == 'MNIST':
            datasets.MNIST(self.data_dir, train=True, download=True)
            datasets.MNIST(self.data_dir, train=False, download=True)
        elif self.dataset_name == 'FashionMNIST':
            datasets.FashionMNIST(self.data_dir, train=True, download=True)
            datasets.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # Load the dataset
            dataset = getattr(datasets, self.dataset_name)(self.data_dir, train=True, transform=self.transform)
            self.dataset_train, self.dataset_val = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.dataset_test = getattr(datasets, self.dataset_name)(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False)


# Transform to dequantize the image so they are continuous
class DequantizeTransform:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        # Get the image as a tensor between 0 and 1
        x = self.transforms(x)

        # Dequantize the image by adding noise and scale to be between -1 and 1
        #x = x + torch.rand_like(x) / 256
        #x = x * 2 - 1
        x = x * 255
        x = x/256 #- 0.5
        x = x + torch.rand_like(x) / 256

        return x


# Example usage
if __name__ == '__main__':
    # Setup the data modeule
    data_dir = '/scratch/pytorch_data/'

    data = DataModule(data_dir, dataset_name='CIFAR10')
    data.prepare_data()
    data.setup('fit')

    # Get the dataset sample
    x, y = data.dataset_train[0]

    # Show the image
    utils.show_imgs(x)




