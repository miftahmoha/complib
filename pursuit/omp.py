from copy import deepcopy
import numpy as np


class OMP:
    def fit(self, X, D, MAX_ITER, EPS):
        row, col = D.shape
        R = deepcopy(X)

        # initialization
        previous = 0
        k = 0
        id_alpha = np.zeros(col)
        Gamma = np.empty((row, 0))
        alpha_dict = {}

        while (
            k < MAX_ITER
            and np.linalg.norm(R - previous) > EPS
            and np.linalg.norm(R) > EPS
        ):
            previous = R
            # computing the inner product of the signal with all dictionary atoms
            M = [abs(np.transpose(D[:, j]) @ R) for j in range(col)]
            # capturing the index of the atom with the highest projection
            l = np.argmax(M)
            # adding the atome to the Gamma matrix
            Gamma = np.hstack([Gamma, np.reshape(D[:, l], (row, 1))])
            # saving the index highest atome to "order" the alpha vector
            alpha_dict[l] = True
            # solving mean-squares minimization for non-indexed alpha (nid_alpha)
            nid_alpha = (
                np.linalg.pinv(np.transpose(Gamma) @ Gamma) @ np.transpose(Gamma) @ X
            )
            # indexing alpha
            for i, j in zip(alpha_dict.keys(), range(0, len(nid_alpha))):
                id_alpha[i] = nid_alpha[j]
            # updating R
            R = X - Gamma @ nid_alpha
            k = k + 1
        return id_alpha
