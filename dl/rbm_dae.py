"""
Improving Autoencoder Performance with Pretrained RBMs

https://towardsdatascience.com/improving-autoencoder-performance
-with-pretrained-rbms-e2e13113c782
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# local imports
from dl.models.dae import DAE_model
from dl.models.train_utils import train_rbm
from dl.models.utils import *


class RBM_DAE:
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
        row, col = X_train.shape

        DL_X_train = DataLoader(
            TensorDataset(torch.Tensor(X_train).to(DEVICE)),
            batch_size=batch_size,
            shuffle=False,
        )

        DL_X_train_RBM = DL_X_train
        # pretrain RBMs
        visible_dim = col
        models = []  # trained RBM models
        for _, hidden_dim in enumerate(dims):
            # hyperparameters
            hidden_dim = hidden_dim
            num_epochs = num_epochs
            lr = 1e-3 if (_ == len(dims) - 1) else 1e-1
            use_gaussian = True if (_ == len(dims) - 1) else False

            # train RBM
            print(f"{visible_dim} to {hidden_dim}")
            model, v, v_pred = train_rbm(
                DL_X_train_RBM,
                visible_dim,
                hidden_dim,
                k=1,
                num_epochs=num_epochs,
                lr=lr,
                use_gaussian=use_gaussian,
            )
            models.append(model)

            # rederive new data loader based on hidden activations of trained model
            new_data = []
            for data_list in DL_X_train_RBM:
                p = model.sample_h(data_list[0])[0]
                new_data.append(p.detach().cpu().numpy())
            new_input = np.concatenate(new_data)
            DL_X_train_RBM = DataLoader(
                TensorDataset(torch.Tensor(new_input).to(DEVICE)),
                batch_size=batch_size,
                shuffle=False,
            )

            # update new visible_dim for next RBM
            visible_dim = hidden_dim

        # fine-tune Deep Autoencoder from built from RBMs
        lr = lr
        self.dae = DAE_model(models).to(DEVICE)
        loss = nn.MSELoss()
        optimizer = optim.Adam(self.dae.parameters(), lr)
        num_epochs = 50

        # train
        for epoch in range(num_epochs):
            losses = []
            for i, data_list in enumerate(DL_X_train):
                data = data_list[0]
                v_pred = self.dae(data)
                batch_loss = loss(
                    data, v_pred
                )  # difference between actual and reconstructed
                losses.append(batch_loss.item())
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            running_loss = np.mean(losses)
            print(f"Epoch {epoch}: {running_loss}")

    # evaluation
    def eval(self, X_eval):
        self.dae.double()

        n = len(X_eval)
        X_eval = torch.tensor(X_eval, dtype=torch.double).to(DEVICE).reshape(1, n)

        with torch.no_grad():
            X_encoded = self.dae.encode(X_eval)
            X_approx = self.dae(X_eval)

        relative_error = torch.norm(X_eval - X_approx) / torch.norm(X_eval)
        return X_encoded, X_approx, relative_error
