'''
utils.py
- Script with useful functions
'''

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pickle


def show_imgs(imgs, nrow=5, return_grid=False, colormap=None, scale_each=True, normalize=True, **kwargs):
    '''
    Show any type of image
    '''

    # Check if it is a tensor, if not make it a tensor
    imgs = check_type(imgs, 'tensor').detach().cpu()



    # Put the images in a grid and show them
    if 'val_range' in kwargs:
        grid = torchvision.utils.make_grid(imgs, nrow=int(nrow), normalize=normalize, value_range=kwargs['val_range'])

    else:
        grid = torchvision.utils.make_grid(imgs, nrow=int(nrow), scale_each=scale_each, normalize=normalize)

    if not return_grid:
        f = plt.figure()
        f.set_figheight(15)
        f.set_figwidth(15)

        plt.axis("off")

        if colormap is not None:
            plt.imshow(grid[0].unsqueeze(0).permute(1, 2, 0).numpy(), cmap=colormap, vmin=grid.min(), vmax=grid.max())
        else:
            plt.imshow(grid.permute(1, 2, 0).numpy())

        # Use a custom title
        if 'title' in kwargs:
            plt.title(kwargs['title'])
        plt.tight_layout()
        plt.margins(x=0, y=0)
        plt.show()

    else:
        return grid


def check_type(x, dtype):
    '''
    Function to check if x is of type dtype (if not, convert it)
    Check for to see if its array or tensor
    '''
    if dtype == 'tensor':
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
    elif dtype == 'array':
        if not isinstance(x, np.ndarray):
            x = x.numpy()
    return x


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)

    return obj


def write_pickle(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)