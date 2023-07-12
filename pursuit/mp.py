from copy import deepcopy
import numpy as np

# Matching Pursuit


class MP:
    def fit(self, X, D, MAX_ITER, EPS):
        row, col = D.shape
        R = deepcopy(X)

        # initialization
        k = 0
        previous = np.zeros_like(X)
        alpha = np.zeros(col)
        while (
            np.linalg.norm(R - previous) > EPS
            and k < MAX_ITER
            and np.linalg.norm(R) > EPS
        ):
            previous = R
            # computing the inner product of the signal with all dictionary atoms
            M = [
                abs(np.matmul(np.transpose(D[:, j]), R)) / np.linalg.norm(D[:, j])
                for j in range(0, col)
            ]
            # capturing the index of the atom with the highest projection
            l = np.argmax(M)
            # updating alpha
            z = np.dot(D[:, l], R) / (np.linalg.norm(D[:, l])) ** 2
            alpha[l] = alpha[l] + z
            # updating R
            R = R - z * D[:, l]
            k = k + 1
        return alpha
