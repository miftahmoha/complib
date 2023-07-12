from copy import deepcopy
import heapq
import numpy as np


def select_highest_k_indices(lst, k):
    indexed_values = [(i, v) for i, v in enumerate(lst)]
    highest_k = heapq.nlargest(k, indexed_values, key=lambda x: x[1])
    highest_k_indices = [i for i, _ in highest_k]
    return highest_k_indices


#  Compressive Sampling Matched Pursuit


class CoSaMP:
    def fit(self, X, D, MAX_ITER, EPS, s):
        row, col = D.shape
        R = deepcopy(X)

        previous = 0
        k = 0
        Gamma = np.empty((row, 0))
        alpha_dict = {}
        saved_atoms = {}
        while (
            k < MAX_ITER
            and np.linalg.norm(R - previous) > EPS
            and np.linalg.norm(R) > EPS
        ):
            previous = R
            """for j in range(K):
                dict_select[j] = abs((np.transpose(D[:, j]) @ R))"""
            M = [abs(np.dot(D[:, j], R)) for j in range(col)]

            # selecting the "2(s)" atoms with largest projection
            largest_atoms = select_highest_k_indices(M, 2 * s)

            # adding the "2(s)" atoms "Gamma" while avoiding repetition
            for k in largest_atoms:
                if k not in saved_atoms.keys():
                    saved_atoms[k] = True
                    Gamma = np.hstack([Gamma, np.reshape(D[:, k], (row, 1))])

            nid_alpha = (
                np.linalg.pinv(np.transpose(Gamma) @ Gamma) @ np.transpose(Gamma) @ X
            )

            for i, j in zip(saved_atoms, range(0, len(nid_alpha))):
                alpha_dict[i] = nid_alpha[j]

            largest_keys = heapq.nlargest(s, alpha_dict, key=alpha_dict.get)
            id_alpha = np.zeros(col)

            for i in largest_keys:
                id_alpha[i] = alpha_dict[i]

            R = X - D @ id_alpha

            alpha_dict.clear()
            k = k + 1
        return id_alpha
