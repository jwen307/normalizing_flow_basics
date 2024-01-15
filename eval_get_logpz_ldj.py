#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 06:56:12 2022

@author: jeff

train.py
    - Script to train a conditional normalizing flow
"""

import torch
import os
import matplotlib.pyplot as plt

from models.flow import Flow
import datasets
import utils


#Get the checkpoint arguments if you want to resume training
load_ckpt_dir = '/scratch/joseph/normalizing_flows/Flow/version_26/'
load_last_ckpt = False # Set to true if you want to load the last checkpoint


if __name__ == "__main__":

    #Load the previous configurations
    ckpt_name = 'last.ckpt' if load_last_ckpt else 'best.ckpt'
    ckpt = os.path.join(load_ckpt_dir,
                        'checkpoints',
                        ckpt_name)

    #Get the configuration file
    config_file = os.path.join(load_ckpt_dir, 'configs.pkl')
    config = utils.read_pickle(config_file)

    # Get the data
    data = datasets.DataModule(data_dir=config['data_args']['dataset_dir'],
                                dataset_name=config['data_args']['dataset_name'],
                                batch_size=config['train_args']['batch_size'],
                                )
    data.prepare_data()
    # Get the testing data
    data.setup(stage='test')

    #Load the model
    print('Loading checkpoint: {}'.format(ckpt))
    model = Flow.load_from_checkpoint(ckpt, config=config)
    model.cuda()

    # Visualize some samples
    with torch.no_grad():
        z = model.sample_distrib(20, temp=1.0) #Note: results are better with temp=0.7 which GLOW does during evaluation
        samples, _ = model(z, rev=True)

        utils.show_imgs(samples, title='Samples from the model')

    # Get the nll, logpz, and ldj
    batch = next(iter(data.test_dataloader()))
    x, y = batch
    x = x.to(model.device)

    with torch.no_grad():
        z, ldj = model(x, rev=False)

        # Get the NLL
        nll = model.get_nll(z,ldj, reduction=None)

        # Get the logpz and ldj separately
        logpz, ldj = model.get_log_pz_ldj(x, rev=False)

        # Get the logpz and ldj separately for each layer
        logpz_layers, ldj_layers = model.get_log_pz_ldj(x, rev=False, intermediate_outputs=True)

    # Visualize a histogram of nll
    plt.hist(nll.cpu().numpy(), bins=25)
    plt.title('Histogram of NLL')
    plt.xlabel('NLL')
    plt.ylabel('Count')
    plt.show()

    # Visualize the logpz and ldj
    plt.plot(logpz.cpu().numpy(), ldj.cpu().numpy(), '.')
    plt.title('Logpz vs. LDJ')
    plt.xlabel('Logpz')
    plt.ylabel('LDJ')
    plt.show()

    # Visualize the logpz and ldj for each layer
    for i in range(model.num_layers):
        plt.plot(logpz_layers[i].cpu().numpy(), ldj_layers[i].cpu().numpy(), '.')
        plt.title('Logpz vs. LDJ for Layer {}'.format(i))
        plt.xlabel('Logpz')
        plt.ylabel('LDJ')
        plt.show()






       
        

