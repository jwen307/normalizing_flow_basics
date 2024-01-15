#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
op.py
    - Operational modules for the normalizing flow
"""
import FrEIA.modules as Fm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Perform the squeeze operation
class SqueezeFlow(Fm.InvertibleModule):
    
    def __init__(self, dims_in, dims_c=[]):
        super().__init__(dims_in, dims_c)
        
    def forward(self, x, rev=False, jac = True):
        x=x[0]
        b, c, h, w = x.shape
        
        if not rev:
            #Forward operation: h x w x c -> h/2 x w/2 x 4c
            x = x.reshape(b,c, h//2, 2, w//2, 2)
            x = x.permute(0,1,3,5,2,4)
            x = x.reshape(b, 4*c, h//2, w//2)
            
        else:
            #Reverse operation: h/2 x w/2 x 4c -> h x w x c
            x = x.reshape(b,c//4, 2,2,h,w)
            x = x.permute(0,1,4,2,5,3)
            x = x.reshape(b,c//4, h*2, w*2)
            
        return [x,], 0
    
    def output_dims(self, input_dims):
        '''See base class for docstring'''
        #print(input_dims)
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return [(input_dims[0][0] * 4, input_dims[0][1]//2, input_dims[0][2]//2)]


class Invertible1x1Conv(Fm.InvertibleModule):
    # Note: this uses LU decomposition so it is faster

    def __init__(self, dims_in, dims_c=[], layer_type='bijective', gamma=0.0, activation='linear', channel_scale=2.0,
                 LU=True):
        super().__init__(dims_in, dims_c)

        self.layer_type = layer_type
        self.gamma = torch.tensor(gamma).cuda()
        # print(self.gamma)
        self.activation = activation

        self.channels, self.h, self.w = dims_in[0]

        self.channel_scale = channel_scale
        self.LU = LU

        # Build the permutation matrix
        # Glow 1x1 conv if bijective

        '''
        random_mat = torch.randn(self.channels, self.channels)
        #Get the QR decomposition of the matrix
        w_q, _ = torch.linalg.qr(random_mat)
        '''

        random_mat = torch.randn(self.channels, self.channels)
        w_q = torch.linalg.qr(random_mat)[0]

        self.activation = 'linear'

        # Mix between https://github.com/rosinality/glow-pytorch/blob/master/model.py and  https://github.com/swing-research/conditional-trumpets/blob/main/glow_ops.py
        if self.LU:
            # Get the LU decomposition
            lu, pivot = torch.linalg.lu_factor(w_q)
            # print(w_q)

            # Unpack the LU decomposition
            w_p, w_l, w_u = torch.lu_unpack(lu, pivot)
            w_s = torch.diag(w_u)
            sign_s = torch.sign(w_s)
            log_s = torch.log(torch.abs(w_s))
            w_u = torch.triu(w_u, 1)

            u_mask = torch.triu(torch.ones_like(w_u), 1)
            l_mask = u_mask.T

            self.register_buffer('pivot', pivot)
            self.register_buffer("w_p", w_p)
            self.register_buffer("u_mask", u_mask)
            self.register_buffer("l_mask", l_mask)
            self.register_buffer("s_sign", sign_s)
            self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))

            self.w_l = nn.Parameter(w_l.double())
            self.log_s = nn.Parameter(log_s.double())
            self.w_u = nn.Parameter(w_u.double())

        else:

            self.weight = torch.nn.Parameter(w_q.double())



    def calc_LU_weight(self):
        '''
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.log_s)))
            )
        '''

        l = self.w_l * self.l_mask + torch.eye(self.channels).cuda()
        u = ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.log_s)))

        weight = torch.matmul(self.w_p, torch.matmul(l, u))

        return weight, l, u

    def forward(self, x, rev=False, jac=True):

        x = x[0]

        # Get the weight if LU
        if self.LU:
            self.weight, l, u = self.calc_LU_weight()
            log_det_jac = self.h * self.w * torch.sum(self.log_s)

        else:

            # Get the singular values of the matrix
            s = torch.linalg.svdvals(self.weight)

            # Find the logs of the singular values to later calculate the determinant
            log_s = torch.log(s + self.gamma ** 2 / (s + 1e-8))
            # print('logs {0}'.format(log_s))

            log_det_jac = torch.sum(log_s) * self.h * self.w



        # Normalizing direction
        if not rev:
            if self.activation == 'relu':
                x = x[:, :self.channels // 2, :, :] - x[:, self.channels // 2:, :, :]

            # Apply the 1x1 conv (note: padding doesn't matter since kernel is 1x1)
            x = torch.nn.functional.conv2d(x, self.weight.unsqueeze(2).unsqueeze(3).float(), padding='same')

        # Generating direction
        else:

            if self.layer_type == 'bijective' and self.LU:
                lower_upper = l + u - torch.eye(self.channels).cuda()

                # w_inv = torch.lu_solve(torch.eye(self.channels, dtype=lower_upper.dtype).cuda(), lower_upper,
                #                        self.pivot.cuda())
                w_inv = torch.linalg.lu_solve( lower_upper,
                                        self.pivot.cuda(),
                                        torch.eye(self.channels, dtype=lower_upper.dtype).cuda())
                w_inv = w_inv.t()
            else:
                # Calculate the pseudo-inverse
                prefactor = torch.matmul(self.weight, self.weight.t()) + self.gamma ** 2 * torch.eye(
                    self.weight.shape[0]).cuda()
                # print('prefactor {0}'.format(prefactor))

                w_inv = torch.matmul(torch.linalg.inv(prefactor), self.weight)

            if self.activation == 'relu':
                conv_filter = torch.concat([w_inv, -w_inv], dim=1)
                x = torch.nn.functional.conv2d(x, conv_filter.unsqueeze(2).unsqueeze(3).float())
                x = torch.nn.functional.relu(x)

            else:
                x = torch.nn.functional.conv2d(x, w_inv.t().unsqueeze(2).unsqueeze(3).float())

            log_det_jac *= -1.0

        return (x,), log_det_jac

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        # print(input_dims)
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")

        if self.layer_type == 'bijective':
            return input_dims

        else:
            # print(input_dims[0][0]//2)
            return [(int(round(input_dims[0][0] // self.channel_scale)), input_dims[0][1], input_dims[0][2])]



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
            #x, log_det_jac1 = self.dequantize(x)
            x, log_det_jac2 = self.sigmoid(x, rev=True)
            #log_det_jac = log_det_jac1 + log_det_jac2
            log_det_jac = log_det_jac2

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
        #log_det_jac = -np.log(self.num_bins) * np.prod(x.shape[1:])
        log_det_jac = 0

        return x, log_det_jac

    def output_dims(self, input_dims):
        return input_dims


# Function to give pixel values between 0-255
def quantize(x):
    # Assumes x is between 0 and 1
    return (x * 255).to(torch.int32)
    

