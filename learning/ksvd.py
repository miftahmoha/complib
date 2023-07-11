from copy import deepcopy
import numpy as np


class KSVD:
    def fit(self, X, K, solver, EPS, MAX_ITER):
        row, col = X.shape
        Z = deepcopy(X).to_numpy()

        # initialization
        D0 = Z[:, 0:K]
        alpha_matrix = np.eye(K, col)
        alpha_0 = []
        E = Z
        for k in range(K):
            # solving for alpha for each signal in the data X
            for i in range(col):
                alpha_matrix[:, i] = solver.fit(Z[:, i], D0, EPS, MAX_ITER)
            # computing the error without counting the kth atom
            for i in range(K):
                if not (i == k):
                    E = E - D0[:, i].reshape(row, 1) @ alpha_matrix[i, :].reshape(
                        1, col
                    )
                else:
                    pass
            # searching for the number and indices of the examples that use the kth atom
            W = np.count_nonzero(alpha_matrix[k, :])
            I = [i for i, e in enumerate(alpha_matrix[k, :]) if e != 0]
            # constructing the Omega matrix
            Omega = np.zeros((col, W))
            # filling the Omega matrix
            for g in range(0, W):
                Omega[I[g], g] = 1
            # updating E
            updated_E = E @ Omega
            # applying K-SVD
            u, s, vh = np.linalg.svd(updated_E)
            D0[:, k] = u[:, 0]
            # updating alpha_0
            if W == 0:
                alpha_0 = alpha_0 + [k]
                E = Z
                continue
            # updating the non-zero components of alpha_matrix[i, :]
            C = np.zeros((W, 1))
            C[:, 0] = s[0] * vh[:, 0]
            for i, j in zip(I, range(len(C))):
                alpha_matrix[k, :][i] = C[j]
            # reinitializing E
            E = Z
        return D0
