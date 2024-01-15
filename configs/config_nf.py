#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:13:37 2022

@author: jeff
Configuration file for training a conditional flow with a U-Net
"""

class Config():
    def __init__(self):

        self.config = {
            'data_args':
                {
                'dataset_dir': '/scratch/joseph/data',
                'dataset_name': 'CIFAR10', # Either 'CIFAR10', 'MNIST', 'FashionMNIST'
                # Path to the logging directory
                'log_dir' : "/scratch/joseph/normalizing_flows/"
                },

            'train_args':
                {
                'lr': 1e-3,
                'epochs': 150, #Number of epochs to train for
                'batch_size': 128,
                'pretrain_unet': False
                },

            'flow_args':
                {
                'distribution': 'gaussian',
                'build_num': 0, #Choose a build number from network_builds.py

                # Flow parameters
                'num_layers': 3, #For MNIST, use 2, for CIFAR10, use 3
                'downsample': 'squeeze',
                'num_steps': 16, #For MNIST, used 8, for CIFAR10, use 16
                'subnet': 'conv3x3',  # Select a subnetwork type: conv3x3, unet, resnet
                'permutation': 'conv1x1',  # Select a permutation type: 'conv1x1, 'fixed1x1', 'random'
                'num_hidden_layers' : [256,256,256], #Number of hidden layers for each subnet

                'use_fc_block': False,
                'num_fc_blocks': 2,
                'cond_fc_size': 64,

                },

        }



