"""
Improving Autoencoder Performance with Pretrained RBMs

https://towardsdatascience.com/improving-autoencoder-performance
-with-pretrained-rbms-e2e13113c782

Eugene Tang

------------------------------------------------------------------

A set of generically helpful utility methods and constants
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Which device to train the model on"""


def seed_everything(seed=42):
    """
    Seed everything to make the code more reproducable.

    This code is the same as that found from many public Kaggle kernels.

    :param seed: seed value to ues
    :type seed: int
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def display_output(v0, vk, dim1=28, dim2=28, v0_fname=None, vk_fname=None):
    """
    Dsplaying the original and reconstructed images for comparison

    :param v0: the original image
    :type v0: Tensor
    :param vk: the reconstructed image
    :type vk: Tensor
    :param dim1: number of pixels on first dimension for plotting
    :type dim1: int
    :param dim2: number of pixels on second dimension for plotting
    :type dim2: int
    :param v0_fname: filename to save plot of original image in
    :type v0_fname: str
    :param vk_fname: filename to save plot of reconstructed image in
    :type vk_fname: str
    """

    print("Original (top) and Reconstructed (bottom)")
    img = make_grid(v0.view(v0.shape[0], 1, dim1, dim2).data)
    npimg = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
    plt.imshow(npimg)
    if v0_fname is not None:
        plt.savefig(v0_fname)
    plt.show()
    img = make_grid(vk.view(vk.shape[0], 1, dim1, dim2).data)
    npimg = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
    plt.imshow(npimg)
    if vk_fname is not None:
        plt.savefig(vk_fname)
    plt.show()


def display_2d_repr(data, labels, fname=None):
    """
    Display a 2d representation of the MNIST digits

    :param data: 2d representation of MNIST digits
    :type data: Tensor
    :param labels: the label for each data point in data
    :type labels: list
    :param frane:filename to save plot in
    :type frame: str
    """

    digit_to_color = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    xs = np.array([x[0] for x in data])
    ys = np.array([x[1] for x in data])

    fig, ax = plt.subplots()
    labels_to_show = labels[0 : len(data)]
    for digit in range(10):
        ix = np.where(labels_to_show == digit)
        ax.scatter(xs[ix], ys[ix], c=digit_to_color[digit], label=digit, marker=".")
    ax.legend()
    if fname is not None:
        plt.savefig(fname)
    plt.show()
