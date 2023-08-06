import numpy as np
import heapq


def select_highest_k_indices(lst, k):
    indexed_values = [(i, v) for i, v in enumerate(lst)]
    highest_k = heapq.nlargest(k, indexed_values, key=lambda x: x[1])
    highest_k_indices = [i for i, _ in highest_k]
    return highest_k_indices


#  Compressive Sampling Matched Pursuit


class CoSaMP:
    """
    CoSaMP algorithm for sparse representation.

    :param X: signal to be represented.
    :type X: numpy.ndarray
    :param D: dictionary matrix with atoms as columns
    :type D: numpy.ndarray
    :param MAX_ITER: maximum number of iterations
    :type MAX_ITER: int
    :param EPS: threshold for convergence
    :type EPS: float
    :param s: desired sparsity level
    :type s: float

    :return: sparse representation of the signal using the dictionary
    :rtype: numpy.ndarray
    """

    def fit(self, X, D, MAX_ITER, EPS, s):
        row, col = D.shape
        R = X.copy()

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
            M = np.abs(D.T @ R)

            # selecting the 2*s atoms with largest projection
            largest_atoms = select_highest_k_indices(M, 2 * s)

            # adding the 2*s atoms while avoiding repetition
            for k_val in largest_atoms:
                if k_val not in saved_atoms.keys():
                    saved_atoms[k_val] = True
                    Gamma = np.hstack([Gamma, np.reshape(D[:, k_val], (row, 1))])

            nid_alpha = np.linalg.pinv(Gamma.T @ Gamma) @ Gamma.T @ X

            alpha_dict = {
                i: nid_alpha[j] for i, j in zip(saved_atoms, range(0, len(nid_alpha)))
            }

            largest_keys = heapq.nlargest(s, alpha_dict, key=alpha_dict.get)
            id_alpha = np.zeros(col)

            for i in largest_keys:
                id_alpha[i] = alpha_dict[i]

            R = X - D @ id_alpha

            alpha_dict.clear()
            k += 1
        return id_alpha
