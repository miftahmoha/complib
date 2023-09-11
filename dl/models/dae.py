"""
Improving Autoencoder Performance with Pretrained RBMs

https://towardsdatascience.com/improving-autoencoder-performance
-with-pretrained-rbms-e2e13113c782

Eugene Tang

------------------------------------------------------------------

Implementation of a Deep Autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .display_utils import *


class DAE_model(nn.Module):
    """
    A Deep Autoencoder that takes a list of RBMs as input
    """

    def __init__(self, models):
        """
        Create a deep autoencoder based on a list of RBM models

        :param models: a list of RBM models to use for autoencoding
        :type models: list[RBM]
        """

        super(DAE_model, self).__init__()

        # extract weights from each model
        encoders = []
        encoder_biases = []
        decoders = []
        decoder_biases = []
        for model in models:
            encoders.append(nn.Parameter(model.W.clone()))
            encoder_biases.append(nn.Parameter(model.h_bias.clone()))
            decoders.append(nn.Parameter(model.W.clone()))
            decoder_biases.append(nn.Parameter(model.v_bias.clone()))

        # build encoders and decoders
        self.encoders = nn.ParameterList(encoders)
        self.encoder_biases = nn.ParameterList(encoder_biases)
        self.decoders = nn.ParameterList(reversed(decoders))
        self.decoder_biases = nn.ParameterList(reversed(decoder_biases))

    def forward(self, v):
        """
        Forward step

        :param v: input tensor
        :type v: Tensor

        :return: reconstruction of v from the autoencoder
        :rtype: Tensor
        """

        # encode
        p_h = self.encode(v)

        # decode
        p_v = self.decode(p_h)

        return p_v

    def encode(self, v):  # for visualization, encode without sigmoid
        """
        Encode input

        :param v: visible input tensor
        :type v: Tensor

        :return: activations of the last layer
        :rtype: Tensor
        """

        p_v = v
        activation = v
        for i in range(len(self.encoders)):
            W = self.encoders[i]
            h_bias = self.encoder_biases[i]
            activation = torch.mm(p_v, W) + h_bias
            p_v = torch.sigmoid(activation)

        # for the last layer, we want to return the activation directly rather than the sigmoid
        return activation

    def decode(self, h):
        """
        Encode hidden layer

        :param h: activations from last hidden layer
        :type h: Tensor

        :return: reconstruction of original input based on h
        :rtype: Tensor
        """

        p_h = h
        for i in range(len(self.encoders)):
            W = self.decoders[i]
            v_bias = self.decoder_biases[i]
            activation = torch.mm(p_h, W.t()) + v_bias
            p_h = torch.sigmoid(activation)
        return p_h


class Naive_DAE_model(nn.Module):
    """A Naive implementation of the DAE_model to be trained without RBMs"""

    def __init__(self, layers):
        """
        Initialize the DAE_model

        :param layers: number of dimensions in each layer of the DAE_model
        :type layers: list[int]
        """

        super(Naive_DAE_model, self).__init__()

        self.layers = layers
        encoders = []
        decoders = []
        prev_layer = layers[0]
        for layer in layers[1:]:
            encoders.append(nn.Linear(in_features=prev_layer, out_features=layer))
            decoders.append(nn.Linear(in_features=layer, out_features=prev_layer))
            prev_layer = layer
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

    def forward(self, x):
        """
        Forward step

        :param x: input tensor
        :type x: Tensor

        :return: reconstructed version of x
        :rtype: Tensor
        """

        x_encoded = self.encode(x)
        x_reconstructed = self.decode(x_encoded)
        return x_reconstructed

    def encode(self, x):
        """
        Encode the input x

        :param x: input to encode
        :type x: Tensor

        :return: encoded input
        :rtype: Tensor
        """

        for i, enc in enumerate(self.encoders):
            if i == len(self.encoders) - 1:
                x = enc(x)
            else:
                x = torch.sigmoid(enc(x))
        return x

    def decode(self, x):
        """
        Decode the representation x

        :param x: input to decode
        :type x: Tensor

        :return: decoded input
        :rtype: Tensor
        """

        for dec in self.decoders:
            x = torch.sigmoid(dec(x))
        return x
