#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:18:41 2022

@author: jeff
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Callable, Union
import pytorch_lightning as pl
from pytorch_lightning import loggers

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import utils



def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(256, c_out, 3, padding=1))


def subnet_res(c_in, c_out, c_hidden):
    return GatedConvNet(c_in, c_hidden, c_out)


class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def forward(self, x):
        return torch.cat([F.elu(x), F.elu(-x)], dim=1)


class LayerNormChannels(nn.Module):

    def __init__(self, c_in):
        """
        This module applies layer norm across channels in an image. Has been shown to work well with ResNet connections.
        Inputs:
            c_in - Number of channels of the input
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_in)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class GatedConv(nn.Module):

    def __init__(self, c_in, c_hidden):
        """
        This module applies a two-layer convolutional ResNet block with input gate
        Inputs:
            c_in - Number of channels of the input
            c_hidden - Number of hidden dimensions we want to model (usually similar to c_in)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            ConcatELU(),
            nn.Conv2d(2 * c_hidden, 2 * c_in, kernel_size=1)
        )

    def forward(self, x):
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)


class GatedConvNet(nn.Module):

    def __init__(self, c_in, c_hidden=32, c_out=-1, num_layers=3):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Inputs:
            c_in - Number of input channels
            c_hidden - Number of hidden dimensions to use within the network
            c_out - Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers - Number of gated ResNet blocks to apply
        """
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        layers = []
        layers += [nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1)]
        for layer_index in range(num_layers):
            layers += [GatedConv(c_hidden, c_hidden),
                       LayerNormChannels(c_hidden)]
        layers += [ConcatELU(),
                   nn.Conv2d(2 * c_hidden, c_out, kernel_size=3, padding=1)]
        self.nn = nn.Sequential(*layers)

        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x):
        return self.nn(x)



class Dequantization(Fm.InvertibleModule):
    '''
    Code adapted from https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial11/NF_image_modeling.ipynb
    Use to dequantize the values of an image
    '''

    def __init__(self, dims_in, alpha=1e-5, num_bins=256, requantize=False):
        '''

        '''
        super().__init__(dims_in)
        self.alpha = alpha
        self.num_bins = num_bins
        self.requantize = requantize

    def forward(self, x, rev=False, jac=True):
        x = x[0]

        # Forward (Normalizing) direction (X -> z)
        # Dequantize, scale to [0,1), and take logit
        # Logit is accounted for in the calcuation of the log det jac
        if not rev:
            x, log_det_jac1 = self.dequantize(x)
            x, log_det_jac2 = self.sigmoid(x, rev=True)
            log_det_jac = log_det_jac1 + log_det_jac2

        # Generating direction (z->X)
        else:
            x, log_det_jac = self.sigmoid(x, rev=False)

            if self.requantize:
                # Rescale to between 0-255
                x = x * self.num_bins
                # Rediscretize
                x = torch.floor(x.clamp(min=0, max=self.num_bins - 1)).to(torch.int32)

                log_det_jac += np.log(self.num_bins) * np.prod(x.shape[1:])

        return (x,), log_det_jac

    def sigmoid(self, x, rev=False):
        '''
        Provides an invertible sigmoid mapping
        '''

        # This is the generating direction (z->X). Reverse of typical notation
        if not rev:
            log_jac_det = (-x - 2 * F.softplus(-x)).sum(
                dim=[1, 2, 3])  # Softplus is a smooth approximation of the ReLU function
            x = torch.sigmoid(x)

        # This is the normalizaling direction (X -> z)
        else:
            # Scale the input to prevent boundaries at 0 and 1 (as proposed in RealNVP)
            x = x * (1 - self.alpha) + 0.5 * self.alpha

            # Calculate the log determinant of the (inverse) Jacobian
            log_jac_det = np.log(1.0 - self.alpha) * np.prod(x.shape[1:])
            log_jac_det += (-torch.log(x) - torch.log(1 - x)).sum(dim=[1, 2, 3])

            # The logit or inverse sigmoid
            x = torch.log(x) - torch.log(1 - x)

        return x, log_jac_det

    def dequantize(self, x):
        '''
        Function to make discrete representations of images into continuous representations.
        This function assumes the values of pixels ranges from 0-255

        '''

        # Transform the discrete values to continuous values
        x = x.to(torch.float32)

        # Add random noise between [0,1/num_bins)
        x = x + torch.rand_like(x).detach()

        # Puts the images between 0 and 1 again
        x = x / self.num_bins

        # This portion is usually added into the loss function for other implementations
        # Accounts for putting the images between 0 and 1 from 0-255
        log_det_jac = -np.log(self.num_bins) * np.prod(x.shape[1:])

        return x, log_det_jac

    def output_dims(self, input_dims):
        return input_dims


# Function to give pixel values between 0-255
def quantize(x):
    # Assumes x is between 0 and 1
    return (x * 255).to(torch.int32)


# Create
# Functions for the mask
def create_checkerboard_mask(h, w, c, invert=False):
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y)
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if invert:
        mask = 1 - mask
    return mask


def create_channel_mask(h, w, c, invert=False):
    mask = torch.cat([torch.ones(c // 2, dtype=torch.float32),
                      torch.zeros(c - c // 2, dtype=torch.float32)])
    mask = mask.view(1, c, 1, 1)
    if invert:
        mask = 1 - mask
    return mask


class RealNVPCouplingLayer(Fm.InvertibleModule):

    def __init__(self, dims_in, dims_c=[], subnet_constructor: Callable = None, num_hidden=32,
                 mask_constructor: Callable = None, invert_mask=False):

        super().__init__(dims_in, dims_c)

        self.channels, self.h, self.w = dims_in[0]

        # Get the selected mask
        mask = mask_constructor(self.h, self.w, self.channels, invert=invert_mask)

        # Store the mask in the register buffer since its not a parameter but should be a module state
        self.register_buffer('mask', mask)

        # Twice as many outputs because separate outputs for s and t
        self.subnet = subnet_constructor(self.channels, 2 * self.channels, num_hidden)

        # Scaling factor to allow different limits
        self.scale_factor = torch.nn.Parameter(torch.zeros(self.channels))

    def forward(self, x, rev=False, jac=True):

        # x is passed as a list, so use x[0]
        x = x[0]

        # Apply the mask
        x_mask = x * self.mask

        # Pass the masked version in
        s, t = self.subnet(x_mask).chunk(2, dim=1)

        # Apply the scaling factor (learnable) for stability
        s_fac = self.scale_factor.exp().view(1, -1, 1, 1)

        # print(s_fac.size())
        # print(s.size())
        s = torch.tanh(s / s_fac) * s_fac

        # Mask the outputs so you only transform the other portions
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Apply the affine transformation
        if not rev:
            x = (x + t) * torch.exp(s)
            log_det_jac = s.sum(dim=[1, 2, 3])

        else:
            x = (x * torch.exp(-s)) - t
            log_det_jac = - s.sum(dim=[1, 2, 3])

        return (x,), log_det_jac

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return input_dims


class SqueezeFlow(Fm.InvertibleModule):

    def __init__(self, dims_in, dims_c=[]):
        super().__init__(dims_in, dims_c)

    def forward(self, x, rev=False, jac=True):
        x = x[0]
        b, c, h, w = x.shape

        if not rev:
            # Forward operation: h x w x c -> h/2 x w/2 x 4c
            x = x.reshape(b, c, h // 2, 2, w // 2, 2)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.reshape(b, 4 * c, h // 2, w // 2)

        else:
            # Reverse operation: h/2 x w/2 x 4c -> h x w x c
            x = x.reshape(b, c // 4, 2, 2, h, w)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.reshape(b, c // 4, h * 2, w * 2)

        return [x, ], 0

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        # print(input_dims)
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return [(input_dims[0][0] * 4, input_dims[0][1] // 2, input_dims[0][2] // 2)]


# %% PyTorch Lightening Integration

class Flow(pl.LightningModule):

    def __init__(self, flow, import_samples=8, lr=1e-3):
        '''
        flows: FrEIA Invertible Module
        import_samples: number of importance samples to use during testing
        '''
        super().__init__()
        self.flow = flow
        self.import_samples = import_samples
        self.lr = lr

    # rev = False is the normalizing direction, rev = True is generating direction
    def forward(self, x, rev=False):
        z, ldj = self.flow(x, rev=rev)

        return z, ldj

    def get_likelihood(self, imgs, give_bpd=True):
        '''
        For a batch of images, return the likelihood or bits per dimensions (bpd)
        '''

        # Get the latent vectors
        z, ldj = self.flow(imgs)

        # Get the log probability of prior (assuming a Gaussian prior)
        log_pz = -0.5 * torch.sum(z ** 2, 1) - (0.5 * np.prod(z.shape[1:]) * torch.log(torch.tensor(2 * torch.pi)))

        # Get the log likelihood
        log_px = log_pz + ldj
        nll = -log_px

        # Get the bits per dimension if needed
        if give_bpd:
            bpd = nll / (np.prod(z.shape[1:]) * np.log(2))

        return bpd.mean() if give_bpd else log_px

    def configure_optimizers(self):
        # Set up the optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, label = batch

        # Loss is the negative log likelihood in bits per dimension
        loss = self.get_likelihood(x, give_bpd=True)

        # Keep track of the loss
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch

        # Show the validation loss
        with torch.no_grad():
            loss = self.get_likelihood(x, give_bpd=True)

        self.log('val_loss', loss)

        # Show example images
        z = torch.randn(16, np.prod(x.shape[1:])).cuda()

        with torch.no_grad():
            samples, _ = self.flow(z, rev=True)

        samples_grid = utils.show_imgs(samples.float().detach().cpu(), return_grid=True)

        # Show the resulting images
        board = self.logger.experiment
        if batch_idx == 0:
            board.add_image('Val Image', samples_grid, self.current_epoch)

    def test_step(self, batch, batch_idx):
        # Perform importance sampling during testing => estimate likelihood M times for each image (since adding noise to discrete images)
        samples = []
        for _ in range(self.import_samples):
            img_ll = self.get_likelihood(batch[0], give_bpd=False)
            samples.append(img_ll)
        img_ll = torch.stack(samples, dim=-1)

        # To average the probabilities, we need to go from log-space to exp, and back to log.
        # Logsumexp provides us a stable implementation for this
        img_ll = torch.logsumexp(img_ll, dim=-1) - np.log(self.import_samples)

        # Calculate final bpd
        bpd = -img_ll / (np.log(2) * np.prod(batch[0].shape[1:]))
        bpd = bpd.mean()

        self.log('test_bpd', bpd)


def print_num_params(model):
    num_params = sum([np.prod(p.shape) for p in model.parameters()])
    print("Number of parameters: {:,}".format(num_params))


# %% Check to make sure both methods of dequantization are the same (They are)
'''
x = dataset[0][0]
#x = quantize(x).unsqueeze(0)
x = x.unsqueeze(0)

#deq0 = DequantizationEx()
deq1 = Dequantization([1,28,28])

#z0, ldj0 = deq0(x, torch.zeros(1))
z1, ldj1 = deq1(x)

#Check the reverse
#z = torch.randn_like(z1)
#x0, ldj3 = deq0(z, torch.zeros(1), reverse=True)
x1, ldj4 = deq1(z1,rev=True)

#x = preprocess_img(dataset[0][0])
'''

dataset_root = '../datasets/'

dataset = torchvision.datasets.MNIST(root=dataset_root, train=True, transform=transforms.Compose([
    transforms.ToTensor(), quantize]), download=True)
'''
class_index = []
for index, sample in enumerate(dataset):
    #If the sample label is one we want to include, add it to the list
    if sample[-1] in [5]:
        class_index.append(index)


subset = torch.utils.data.Subset(dataset, class_index)
'''
trainset, valset = torch.utils.data.random_split(dataset, [round(len(dataset) * 0.9), round(len(dataset) * 0.1)])

testset = torchvision.datasets.MNIST(root=dataset_root, train=False, transform=transforms.Compose([
    transforms.ToTensor(), quantize]), download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=4)

# %% Construct the flow
nodes = [Ff.InputNode(1, 28, 28, name='Input')]
nodes.append(Ff.Node(nodes[-1], Dequantization, {'alpha': 1e-5, 'num_bins': 256}))
outputs = []
hidden = [32, 32, 32]
mask_constructor = create_channel_mask

for scale in range(1):
    nodes.append(Ff.Node(nodes[-1], SqueezeFlow, {}, name='Scale {0}, Squeeze'))
    nodes.append(Ff.Node(nodes[-1].out0, RealNVPCouplingLayer,
                         {'subnet_constructor': subnet_res, 'num_hidden': hidden[0],
                          'mask_constructor': mask_constructor, 'invert_mask': False},
                         name='Scale {0}, coupling 0'.format(scale)))
    nodes.append(Ff.Node(nodes[-1], RealNVPCouplingLayer,
                         {'subnet_constructor': subnet_res, 'num_hidden': hidden[0],
                          'mask_constructor': mask_constructor, 'invert_mask': True},
                         name='Scale {0}, coupling 1'.format(scale)))
    # nodes.append(Ff.Node(nodes[-1], RealNVPCouplingLayer,
    #                      {'subnet_constructor': subnet_res, 'num_hidden': hidden,
    #                       'mask_constructor': mask_constructor, 'invert_mask': False},
    #                      name='Scale {0}, coupling 2'.format(scale)))

    nodes.append(Ff.Node(nodes[-1], RealNVPCouplingLayer,
                         {'subnet_constructor': subnet_res, 'num_hidden': hidden[1],
                          'mask_constructor': mask_constructor, 'invert_mask': False},
                         name='Scale {0}, coupling 3'.format(scale)))
    nodes.append(Ff.Node(nodes[-1], RealNVPCouplingLayer,
                         {'subnet_constructor': subnet_res, 'num_hidden': hidden[1],
                          'mask_constructor': mask_constructor, 'invert_mask': True},
                         name='Scale {0}, coupling 4'.format(scale)))
    # nodes.append(Ff.Node(nodes[-1], RealNVPCouplingLayer,
    #                      {'subnet_constructor': subnet_res, 'num_hidden': hidden,
    #                       'mask_constructor': mask_constructor, 'invert_mask': False},
    #                      name='Scale {0}, coupling 5'.format(scale)))

    # Store the split portion
    split = Ff.Node(nodes[-1], Fm.Split, {}, name='Scale {0}, Split'.format(scale))
    flatten_split = Ff.Node(split.out1, Fm.Flatten, {}, name='Scale {0} flatten'.format(scale))
    outputs.append(flatten_split.out0)
    nodes.append(flatten_split)  # The flatten portion still needs to be in the graph

    nodes.append(split)
    hidden = hidden * 2

# Last layer
nodes.append(Ff.Node(nodes[-1].out0, SqueezeFlow, {}, name='Scale {0}, Squeeze'))
nodes.append(Ff.Node(nodes[-1], RealNVPCouplingLayer,
                     {'subnet_constructor': subnet_res, 'num_hidden': hidden[2], 'mask_constructor': mask_constructor,
                      'invert_mask': False},
                     name='Scale {0}, coupling 0'.format(3)))
nodes.append(Ff.Node(nodes[-1], RealNVPCouplingLayer,
                     {'subnet_constructor': subnet_res, 'num_hidden': hidden[2], 'mask_constructor': mask_constructor,
                      'invert_mask': True},
                     name='Scale {0}, coupling 1'.format(3)))
nodes.append(Ff.Node(nodes[-1], RealNVPCouplingLayer,
                     {'subnet_constructor': subnet_res, 'num_hidden': hidden[2], 'mask_constructor': mask_constructor,
                      'invert_mask': False},
                     name='Scale {0}, coupling 2'.format(3)))
nodes.append(Ff.Node(nodes[-1], RealNVPCouplingLayer,
                     {'subnet_constructor': subnet_res, 'num_hidden': hidden[2], 'mask_constructor': mask_constructor,
                      'invert_mask': True},
                     name='Scale {0}, coupling 3'.format(3)))

# Flatten output and concatenate with other splits
nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='Scale {0} flatten'.format(3)))
outputs.append(nodes[-1].out0)
nodes.append(Ff.Node(outputs, Fm.Concat, {'dim': 0}, name='concat'))
nodes.append(Ff.OutputNode(nodes[-1], name='output'))

flow = Ff.GraphINN(nodes)
model = Flow(flow, import_samples=8, lr=1e-3)
# model = Flow.load_from_checkpoint('logs/default/mnist_200epochs/checkpoints/epoch=200.ckpt', flow, import_samples = 8, lr=1e-3)
# ckpt = torch.load('logs/default/mnist_200epochs/checkpoints/epoch=200.ckpt')
# model.load_state_dict(ckpt['state_dict'])

# %% Training

#Create the tensorboard logger
log_dir = "/scratch/joseph/normalizing_flows/"
Path(log_dir).mkdir(parents=True, exist_ok=True)
logger = loggers.TensorBoardLogger(log_dir, name='Flow')

#Create the trainer
trainer = pl.Trainer(max_epochs = 201, gradient_clip_val = 1.0,  logger = logger)

#Train the model
print('Starting Training')
trainer.fit(model, trainloader, valloader)

#Get the test results
test_result = trainer.test(model, test_dataloaders=testloader, verbose=False)
print('Results: {0}'.format(test_result))


