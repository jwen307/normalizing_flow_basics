#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:11:18 2022

@author: jeff

flow.py
    - PyTorch Lighting module for a base conditional flow
"""
import torch
import numpy as np
import pytorch_lightning as pl
import torch.optim as optim

import sys
sys.path.append('../')
from models import network_builds
from models.components.inn_modules import misc
import utils


class Flow(pl.LightningModule):
    
    def __init__(self, config):

        '''
        configs: Configurations for the model
        '''
        super().__init__()
        
        self.save_hyperparameters()

        # Figure out the size of the inputs
        if config['data_args']['dataset_name'] == 'CIFAR10':
            img_size = 32
            self.input_dims = [3, img_size, img_size]
        else:
            img_size = 28
            self.input_dims = [1, img_size, img_size]

        # Set the distribution
        self.distrib = config['flow_args']['distribution']

        # Get the latent size
        self.latent_size = np.prod(self.input_dims)

        # Options for builds ( Add more as needed )
        builds = [
            network_builds.build0,
        ]

        # Build the network
        self.build_bij_func = builds[config['flow_args']['build_num']]
        self.config = config

        self.num_layers = self.config['flow_args']['num_layers']

        self.build()


    # Function to build the network
    def build(self):
        #Build the bijective network
        self.flow = self.build_bij_func(self.input_dims, **self.config['flow_args'])

        # Initialize the parameters
        self.init_params()

    def init_params(self):
        """
        Initialize the parameters of the model.

        Returns
        -------
        None.

        """
            
        for key, param in self.flow.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = 0.02 * torch.randn(param.data.shape)
                # last convolution in the coeff func
                if len(split) > 3 and split[3][-1] == '4': 
                    param.data.fill_(0.)
                    
                    
    #rev = False is the normalizing direction, rev = True is generating direction
    def forward(self, x, rev=False, intermediate_outputs=False):
        

        #Normalizing direction
        if not rev:
            z, ldj = self.flow(x, rev=rev, intermediate_outputs=intermediate_outputs)
            
        #Generating direction
        else:
            z, ldj = self.flow(x, rev=rev, intermediate_outputs=intermediate_outputs)

        # Sum up the log-determinants until a split. Give the intermediate log det jacs
        if intermediate_outputs and not rev:
            num_layers = self.config['flow_args']['num_layers']

            # Figure out how the code vector is split up between the layers
            break_down = [0] + np.cumsum([self.latent_size // 2 ** k for k in range(1, num_layers)]).tolist()
            break_down = break_down + [self.latent_size]

            split_num = 0
            split_ldjs = []
            running_ldj = 0
            for key in self.flow.node_list:
                if not key in ldj.keys():
                    continue

                # Add the ldj to the running total
                running_ldj += ldj[key].detach().cpu()

                # Check if we're at a split
                if isinstance(key.module, misc.Split):
                    split_ldjs.append(running_ldj)
                    running_ldj = 0
                    split_num += 1

            out_list = [z[out_node, 0] for out_node
                        in (self.flow.in_nodes if rev else self.flow.out_nodes)]
            if len(out_list) == 1:
                z = out_list[0]
                zs_inter = []
                for j in range(num_layers):
                    zs_inter.append(z[:, break_down[j]: break_down[j + 1]])
                # Outputs is a list of the intermediate code vectors and the split log-determinants
                return zs_inter, torch.stack(split_ldjs, dim=0)
            else:
                return tuple(out_list), split_ldjs
            
        return z, ldj
    

    #Get latent vectors according to the specified distributions
    def sample_distrib(self, num_samples, temp=1.0):
        
        if self.distrib == 'radial':
            z = torch.randn(num_samples, self.latent_size, device=self.device)
            z_norm = torch.norm(z, dim=1)
            
            #Get the radius
            r = torch.abs(torch.randn((num_samples,1), device=self.device))
            
            #Normalize the vectors and then multiply by the radius
            z = z/z_norm.view(-1,1)*(r+temp-1.0)
            
        elif self.distrib == 'gaussian':
            z = torch.randn(num_samples, self.latent_size, device=self.device) * temp
            
        else:
            raise NotImplementedError()
            
        return z




    #Get the likelihood for a given set of latent vectors
    def get_nll(self,z, ldj, give_bpd=True, reduction='mean'):
        
        log_pz = self.get_log_pz(z)
            
        if self.training:
            self.log('log_pz', log_pz.mean(), sync_dist=True)
            self.log('ldj', ldj.mean(), sync_dist=True)

        #Get the log likelihood
        log_px = log_pz + ldj 
        nll = -log_px
        
        #Get the bits per dimension if needed
        if give_bpd:
            #bpd = (nll / (np.prod(z.shape[1:]) * np.log(2))
            # Accounts for the discrete nature of the dataset (done in https://github.com/rosinality/glow-pytorch/blob/master/train.py#L82)
            # and explained in https://www.reddit.com/r/MachineLearning/comments/56m5o2/comment/ddxxhhb/
            #bpd = (nll / (np.prod(z.shape[1:]))+np.log(128)) / np.log(2)
            bpd = (nll / (np.prod(z.shape[1:])) + np.log(256)) / np.log(2)

            #print('bpd: {0}'.format(bpd.mean()))
            return bpd.mean() if reduction == 'mean' else bpd
        
        return nll.mean() if reduction == 'mean' else nll




    # Get the log probability density of the prior
    def get_log_pz(self, z):
        if self.distrib == 'gaussian':
            # Get the log probability of prior (assuming a Gaussian prior)
            log_pz = -0.5 * torch.sum(z ** 2, 1) - (0.5 * np.prod(z.shape[1:]) * torch.log(torch.tensor(2 * torch.pi)))

        elif self.distrib == 'radial':
            # Number of dimensions for each z
            n = torch.prod(torch.tensor(z.shape[1:]))

            # Get the log probability of prior (assuming a radial prior)
            log_pz = torch.log(torch.sqrt(2 / (torch.pi * n))) - (n - 1) * torch.log(
                torch.norm(z, dim=1)) - 0.5 * torch.sum(z ** 2, 1)

        return log_pz



    # Function to get log pz and ldj
    def get_log_pz_ldj(self, z, rev = False,intermediate_outputs=False):
        # z can be a code vector or an image

        # For generating direction, go forward and then backward through the flow
        if rev:
            # Get the image from the code vector
            x, _ = self(z,  rev=True, intermediate_outputs=False)

            # Go back through the flow
            z, ldj = self(x, rev=False, intermediate_outputs=intermediate_outputs)

        # For normalizing direction, go forward through the flow
        else:
            z, ldj = self(z, rev=False, intermediate_outputs=intermediate_outputs)


        # Get the log probability of the prior
        if not intermediate_outputs:
            log_pz = self.get_log_pz(z)

        # If you want the intermediate outputs, get the log probability of the prior for each layer
        else:
            log_pzs = []
            for i in range(len(z)):
                log_pz = self.get_log_pz(z[i])
                log_pzs.append(log_pz)

            log_pz = torch.stack(log_pzs)

        return log_pz, ldj


    # Training setup
    def configure_optimizers(self):

        opt = torch.optim.Adam(self.parameters(),
            lr=self.config['train_args']['lr']
            )

        scheduler = optim.lr_scheduler.StepLR(opt, 1, gamma=0.99)


        return [opt], [scheduler]


    def training_step(self, batch, batch_idx):

        # Get all the inputs
        x = batch[0].to(self.device)

        # Pass through the CNF
        z, ldj = self(x, rev=False)

        # Find the negative log likelihood
        loss = self.get_nll(z, ldj, give_bpd=True)

        # Log the training loss
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x = batch[0]

        # Define the logger to show the image
        board = self.logger.experiment

        with torch.no_grad():
            # Pass through the CNF
            z, ldj = self(x, rev=False)

            # Find the negative log likelihood
            loss = self.get_nll(z, ldj, give_bpd=True)

            self.log('val_loss', loss, sync_dist=True)

            # Show some example images
            if batch_idx == 1:
                # Show the original images and sampled images
                gt_grid = utils.show_imgs(x.float().detach().cpu(), return_grid=True)
                board.add_image('GT Images', gt_grid, self.current_epoch)

                # Get samples from the normalizing flow
                z_samples = self.sample_distrib(10)
                samples = self(z_samples, rev=True)[0]
                samples_grid = utils.show_imgs(samples.float().detach().cpu(), return_grid=True)

                board.add_image('Val Image', samples_grid, self.current_epoch)
    



#%% Test out the model

if __name__ == '__main__':
    from configs.config_nf import Config
    import datasets

    # Get the configurations
    config = Config().config

    # Get the data
    data = datasets.DataModule(data_dir=config['data_args']['dataset_dir'],
                               dataset_name=config['data_args']['dataset_name'],
                               batch_size=config['train_args']['batch_size'],
                               )
    data.prepare_data()
    data.setup()

    model = Flow(config)
    model.cuda()

    x = data.dataset_train[0][0].unsqueeze(0).to(model.device)

    with torch.no_grad():
        z,ldj = model(x, rev=False)
        x_recon = model(z, rev=True)[0]

    #Check invertibility
    print('Reconstruction error: {0}'.format(torch.norm(x-x_recon)))
    utils.show_imgs(x.detach().cpu())
    utils.show_imgs(x_recon.detach().cpu())
