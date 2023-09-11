"""
Improving Autoencoder Performance with Pretrained RBMs

https://towardsdatascience.com/improving-autoencoder-performance
-with-pretrained-rbms-e2e13113c782

Eugene Tang
"""

import torch

from .dae import DAE_model, Naive_DAE_model
from .rbm import RBM
from .display_utils import *


def train_rbm(train_dl, visible_dim, hidden_dim, k, num_epochs, lr, use_gaussian=False):
    """
    Create and train an RBM

    Uses a custom strategy to have 0.5 momentum before epoch 5 and 0.9 momentum after

    :param train_dl: training data loader
    :type train_dl: DataLoader
    :param visible_dim: number of dimensions in visible (input) layer
    :type visible_dim: int
    :param hidden_dim: number of dimensions in hidden layer
    :type hidden_dim: int
    :param k: number of iterations to run for Gibbs sampling (often 1 is used)
    :type k: int
    :param num_epchs: number of epochs to run for
    :type num_epochs: int
    :param lr: learning rate
    :type lr: float
    :param use_gaussian: whether to use a Gaussian distribution for the hidden state
    :type use_gaussian: bool

    :return: a trained RBM model, sample input tensor,
        reconstructed activation probabilities for sample input tensor
    :rtype: RBM object, Tensor, Tensor
    """

    rbm = RBM(
        visible_dim=visible_dim,
        hidden_dim=hidden_dim,
        gaussian_hidden_distribution=use_gaussian,
    )
    loss = torch.nn.MSELoss()  # we will use MSE loss

    for epoch in range(num_epochs):
        train_loss = 0
        for i, data_list in enumerate(train_dl):
            sample_data = data_list[0].to(DEVICE)
            v0, pvk = sample_data, sample_data

            # Gibbs sampling
            for i in range(k):
                _, hk = rbm.sample_h(pvk)
                pvk = rbm.sample_v(hk)

            # compute ph0 and phk for updating weights
            ph0, _ = rbm.sample_h(v0)
            phk, _ = rbm.sample_h(pvk)

            # update weights
            rbm.update_weights(
                v0,
                pvk,
                ph0,
                phk,
                lr,
                momentum_coef=0.5 if epoch < 5 else 0.9,
                weight_decay=2e-4,
                batch_size=sample_data.shape[0],
            )

            # track loss
            train_loss += loss(v0, pvk)

        # print training loss
        print(f"epoch {epoch}: {train_loss/len(train_dl)}")
    return rbm, v0, pvk
