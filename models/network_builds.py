#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network_builds.py
    - Functions to build the conditional normalizing flow architecture
"""

#%%
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import sys
sys.path.append('../')

from models.components.inn_modules import misc, op
from models.components import buildingblocks as nets
from models.components.subnets.subnetworks import subnet_conv3x3, subnet_unet, subnet_res

# Add new subnets here
subnet_dict = {
       'conv3x3': subnet_conv3x3,
       'unet': subnet_unet,
       'resnet': subnet_res,
       }

# Function to build the flow
def build0(img_size=[3, 32, 32], **kwargs):
    """
        Args:
            img_size=[16,320,320]
        Kwargs:
            num_downsample = 3
            cond_conv_chs = [64,64,128]
            downsample='squeeze'
            num_blocks = 20
            use_fc_block=False,
            num_fc_blocks=2,
            cond_fc_size=64
    """

    # Collect all the split nodes
    split_nodes = []

    # Build the flow
    nodes = [Ff.InputNode(img_size[0], img_size[1], img_size[2], name='Input')]

    # Dequantization step (dequantize is done in dataset but sigmoid is done here
    nodes.append(Ff.Node(nodes[-1], op.Dequantization, {'alpha': 1e-5, 'num_bins': 256}))

    # Define each layer
    for k in range(kwargs['num_layers']):

        # 1) Downsample/Squeeze
        nets._add_downsample(nodes, kwargs['downsample'])

        # 2) Add flow steps (Activation Normalizaton, Permutation, Coupling)
        nets._add_flow_step(nodes,
                            downsampling_level=k,
                            num_steps=kwargs['num_steps'],
                            coupling_type='affine',
                            act_norm=True,
                            permutation_type=kwargs['permutation'],
                            subnet_in=subnet_dict[kwargs['subnet']],
                            num_hidden_layers=kwargs['num_hidden_layers'][k],)

        # 3) Split
        nodes.append(Ff.Node(nodes[-1],
                             misc.Split,
                             {},
                             name='split_{}'.format(k)
                             ))
        split_nodes.append(Ff.Node(nodes[-1].out1,
                                   Fm.Flatten,
                                   {},
                                   name='flatten_split_{}'.format(k)))

    # Flatten
    nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {}, name='conv_flatten'))

    # Add the fully connected block if needed
    if kwargs['use_fc_block']:
        nodes.append(Ff.Node(nodes[-1], misc.Split,
                             {'section_sizes': [128], 'dim': 0, 'n_sections': 2},
                             name='split_fc'))

        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten,
                                   {},
                                   name='flatten_split_fc'))

        nets._add_fc_section(nodes,
                             num_blocks=kwargs['num_fc_blocks'],
                             coupling_type='affine')

        nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {}, name='final_flatten'))

    # Concatenate all of the split nodes
    nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                         Fm.Concat1d, {'dim': 0}, name='concat'))

    nodes.append(Ff.OutputNode(nodes[-1], name='out'))

    # Create the flow network
    flow = Ff.GraphINN(nodes +  split_nodes, verbose=False)

    return flow
