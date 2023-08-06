from copy import deepcopy
import numpy as np

# Stagewise Orthogonal Matching Pursuit


class StOMP:
    def fit(self, X, D, MAX_ITER, EPS, t):
        """
        Stagewise Orthogonal Matching Pursuit algorithm for sparse representation.

        :param X: signal to be represented.
        :type X: numpy.ndarray
        :param D: dictionary matrix with atoms as columns
        :type D: numpy.ndarray
        :param MAX_ITER: maximum number of iterations
        :type MAX_ITER: int
        :param EPS: threshold for convergence.
        :type EPS: float
        :param t: ?
        :type t: float

        :return: sparse representation of the signal using the dictionary
        :rtype: numpy.ndarray
        """

        row, col = D.shape
        R = X.copy()

        # initialization
        previous = 0
        k = 0
        Gamma = np.empty((row, 0))
        alpha_dict = {}

        while (
            k < MAX_ITER
            and np.linalg.norm(R - previous) > EPS
            and np.linalg.norm(R) > EPS
        ):
            previous = R
            threshold = t * np.linalg.norm(R) / np.sqrt(col)
            for j in range(col):
                C = abs(np.dot(D[:, j], R))
                if C > threshold:
                    if j not in alpha_dict.keys():
                        Gamma = np.hstack([Gamma, np.reshape(D[:, j], (row, 1))])
                        alpha_dict[j] = C
            nid_alpha = np.linalg.pinv(Gamma.T @ Gamma) @ Gamma.T @ X
            id_alpha = np.zeros(col)

            for i, j in zip(alpha_dict.keys(), range(0, len(nid_alpha))):
                id_alpha[i] = nid_alpha[j]

            R = X - D @ id_alpha
            for i in range(0, len(nid_alpha)):
                if nid_alpha[i] == 0:
                    del alpha_dict[i]
            k += 1
        return id_alpha
