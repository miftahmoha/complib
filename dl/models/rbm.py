"""
Improving Autoencoder Performance with Pretrained RBMs

https://towardsdatascience.com/improving-autoencoder-performance
-with-pretrained-rbms-e2e13113c782

Eugene Tang

------------------------------------------------------------------

Implementation of a Restricted Boltzmann Machine
"""

import torch
from .display_utils import *


class RBM:
    """
    Implementation of a Restricted Boltzmann Machine

    Note that this implementation does not use Pytorch's nn.Module
    because we are updating the weights ourselves

    """

    def __init__(self, visible_dim, hidden_dim, gaussian_hidden_distribution=False):
        """
        Initialize a Restricted Boltzmann Machine

        :param visible_dim:  number of dimensions in visible (input) layer
        :type visible_dim: int
        :param hidden_dim: number of dimensions in hidden layer
        :type hidden_dim: int
        :param gaussian_hidden_distribution: whether to use a Gaussian distribution
            for the values of the hidden dimension instead of a Bernoulli
        :param gaussian_hidden_distribution: bool
        """

        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.gaussian_hidden_distribution = gaussian_hidden_distribution

        # intialize parameters
        self.W = torch.randn(visible_dim, hidden_dim).to(DEVICE) * 0.1
        self.h_bias = torch.zeros(hidden_dim).to(DEVICE)  # v --> h
        self.v_bias = torch.zeros(visible_dim).to(DEVICE)  # h --> v

        # parameters for learning with momentum
        self.W_momentum = torch.zeros(visible_dim, hidden_dim).to(DEVICE)
        self.h_bias_momentum = torch.zeros(hidden_dim).to(DEVICE)  # v --> h
        self.v_bias_momentum = torch.zeros(visible_dim).to(DEVICE)  # h --> v

    def sample_h(self, v):
        """
        Get sample hidden values and activation probabilities

        :param v: tensor of input from visible layer
        :type v: Tensor
        """

        activation = torch.mm(v, self.W) + self.h_bias
        if self.gaussian_hidden_distribution:
            return activation, torch.normal(activation, torch.tensor([1]).to(DEVICE))
        else:
            p = torch.sigmoid(activation)
            return p, torch.bernoulli(p)

    def sample_v(self, h):
        """
        Get visible activation probabilities

        :param h: Tensor of input from hidden
        :type h: Tensor
        """

        activation = torch.mm(h, self.W.t()) + self.v_bias
        p = torch.sigmoid(activation)
        return p

    def update_weights(
        self, v0, vk, ph0, phk, lr, momentum_coef, weight_decay, batch_size
    ):
        """
        Learning step: update parameters

        Uses contrastive divergence algorithm as described in

        :param v0: initial visible state
        :type v0: Tensor
        :param vk: final visible state
        :type vk: Tensor
        :param ph0: hidden activation probabilities for v0
        :type ph0: Tensor
        :param phk: hidden activation probabilities for vk
        :type phk: Tensor
        :param lr: learning rate
        :type lr: float
        :param momentum_coef: coefficient to use for momentum
        :type momentum_coef: float
        :param weight_decay: coefficient to use for weight decay
        :type weight_decay: float
        :param batch_size: size of each batch
        :type batch_size: int
        """

        self.W_momentum *= momentum_coef
        self.W_momentum += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)

        self.h_bias_momentum *= momentum_coef
        self.h_bias_momentum += torch.sum((ph0 - phk), 0)

        self.v_bias_momentum *= momentum_coef
        self.v_bias_momentum += torch.sum((v0 - vk), 0)

        self.W += lr * self.W_momentum / batch_size
        self.h_bias += lr * self.h_bias_momentum / batch_size
        self.v_bias += lr * self.v_bias_momentum / batch_size

        self.W -= self.W * weight_decay  # L2 weight decay
