#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 06:56:12 2022

@author: jeff

train.py
    - Script to train a conditional normalizing flow
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from pathlib import Path
import traceback
from configs.config_nf import Config
from models.flow import Flow

import datasets
import utils


#Get the checkpoint arguments if you want to resume training
load_ckpt_dir = None
load_last_ckpt = False # Set to true if you want to load the last checkpoint


if __name__ == "__main__":
    
    #Use new configurations if not loading a pretrained model
    if load_ckpt_dir is None:

        #Get the configurations
        config = Config().config
        ckpt=None
    
    #Load the previous configurations if resuming
    else:
        ckpt_name = 'last.ckpt' if load_last_ckpt else 'best.ckpt'
        ckpt = os.path.join(load_ckpt_dir,
                            'checkpoints',
                            ckpt_name)

        #Get the configuration file
        config_file = os.path.join(load_ckpt_dir, 'configs.pkl')
        config = utils.read_pickle(config_file)
    

    try:

        # Get the data
        data = datasets.DataModule(data_dir=config['data_args']['dataset_dir'],
                                    dataset_name=config['data_args']['dataset_name'],
                                    batch_size=config['train_args']['batch_size'],
                                    )
        data.prepare_data()
        data.setup()


        #Load the model
        if ckpt is not None:
            print('Loading checkpoint: {}'.format(ckpt))
            model = Flow.load_from_checkpoint(ckpt, config=config)
        # Create a new model
        else:
            model = Flow(config)

        # Compile the model (Doesn't work if there's complex numbers like in fft2c)
        #model = torch.compile(model)

        # Create the tensorboard logger
        Path(config['data_args']['log_dir']).mkdir(parents=True, exist_ok=True)
        logger = loggers.TensorBoardLogger(config['data_args']['log_dir'], name='Flow')

        # Create the checkpoint callback to save models along the way
        ckpt_callback = ModelCheckpoint(
            save_top_k = 1,
            monitor='val_loss',
            mode = 'min',
            filename='best',
            )

        # Create the trainers
        trainer = pl.Trainer(
            max_epochs=config['train_args']['epochs'],
            gradient_clip_val=1.0,
            accelerator='gpu',
            logger=logger,
            check_val_every_n_epoch=1,
            callbacks=[ckpt_callback],
            #strategy='ddp_find_unused_parameters_true',
        )

        # Save the configurations
        model_path = trainer.logger.log_dir
        Path(model_path).mkdir(parents=True, exist_ok=True)
        config_file = os.path.join(model_path, 'configs.pkl')
        utils.write_pickle(config, config_file)

        # Train the model
        if ckpt is None:
            print("Starting Training")
            trainer.fit(model, data.train_dataloader(), data.val_dataloader())

            #Save the last checkpoint just in case
            trainer.save_checkpoint(os.path.join(model_path,'checkpoints','last.ckpt'))

        else:
            print("Resuming Training")
            trainer.fit(model, data.train_dataloader(), data.val_dataloader(),ckpt_path=ckpt)
            trainer.save_checkpoint(os.path.join(model_path,'checkpoints','last.ckpt'))


    except:

        traceback.print_exc()
       
        

