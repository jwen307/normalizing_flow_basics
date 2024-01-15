#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subnetworks.py
    - Functions to describe the subnetworks used in the coupling layers

Code from https://github.com/jleuschn/cinn_for_imaging/blob/master/cinn_for_imaging/reconstructors/networks/cond_net_mri.py
"""
import torch
import torch.nn as nn

from .subnet_modules.gated_conv import GatedConvNet
from .subnet_modules.unet import UNet

def subnet_res(c_in,c_out, hidden_layers=32, num_layers=2):

    return GatedConvNet(c_in, hidden_layers, c_out, num_layers=num_layers)


def subnet_unet(c_in, c_out, c_hidden = [32,64], num_layers = 3):
    c_hidden = [c_in, 2*c_in]
    return UNet(c_in, c_out, c_hidden)



def subnet_conv3x3(in_ch, out_ch, hidden_layers=128):
    """
    Sub-network with 3x3 2d-convolutions and leaky ReLU activation.
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    Returns
    -------
    torch sequential model
        The sub-network.
    """
    subnet = nn.Sequential(
                nn.Conv2d(in_ch, hidden_layers, 3, padding=1),
                nn.LeakyReLU(), 
                nn.Conv2d(hidden_layers, hidden_layers, 3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_layers, out_ch, 3, padding=1))
    
    #subnet.apply(init_weights)
    #torch.nn.init.zeros_(subnet[-1].weight)

    return subnet


def subnet_conv1x1(in_ch, out_ch, hidden_layers=512):
    """
    Sub-network with 1x1 2d-convolutions and leaky ReLU activation.
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    Returns
    -------
    torch sequential model
        The sub-network.
    """
    subnet = nn.Sequential(
                nn.Conv2d(in_ch, hidden_layers, 1),
                nn.LeakyReLU(), 
                nn.Conv2d(hidden_layers, hidden_layers, 1),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_layers, out_ch, 1))
    
    #subnet.apply(init_weights)
    #torch.nn.init.zeros_(subnet[-1].weight)
    
    return subnet


def subnet_fc(in_ch, out_ch):
    """
    Sub-network with fully connected layers and leaky ReLU activation.
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    Returns
    -------
    torch sequential model
        The sub-network.
    """
    subnet =  nn.Sequential(nn.Linear(in_ch, 2*in_ch), 
                         nn.LeakyReLU(), 
                         nn.Linear(2*in_ch, 2*in_ch),
                         nn.LeakyReLU(),
                         nn.Linear(2*in_ch, out_ch))   
    
    #subnet.apply(init_weights)
    #torch.nn.init.zeros_(subnet[-1].weight)
    
    return subnet


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        #m.bias.data.fill_(0.01)


