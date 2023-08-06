"""
Improving Autoencoder Performance with Pretrained RBMs

https://towardsdatascience.com/improving-autoencoder-performance
-with-pretrained-rbms-e2e13113c782

Eugene Tang
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# local imports
from dl.models.dae import Naive_DAE_model
from dl.models.utils import *


class DAE:
    def train(
        self,
        X_train,
        batch_size=5,
        lr=1e-2,
        num_epochs=10,
        dims=[500, 150, 40],
        shuffle=False,
    ):
        X_train = X_train.T
        DL_X_train = DataLoader(
            TensorDataset(torch.Tensor(X_train).to(DEVICE)),
            batch_size=batch_size,
            shuffle=False,
        )
        row, col = X_train.shape
        dims = [col] + dims

        self.naive_dae = Naive_DAE_model(dims).to(DEVICE)
        loss = nn.MSELoss()
        optimizer = optim.Adam(self.naive_dae.parameters(), lr)
        num_epochs = num_epochs

    # evaluation
    def eval(self, X_eval):
        self.naive_dae.double()
        X_eval = torch.tensor(X_eval, dtype=torch.double).to(DEVICE)

        with torch.no_grad():
            X_encoded = self.naive_dae.encode(X_eval)
            X_approx = self.naive_dae(X_eval)

        relative_error = torch.norm(X_eval - X_approx) / torch.norm(X_eval)
        return X_encoded, X_approx, relative_error
