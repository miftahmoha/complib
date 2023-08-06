import numpy as np

# Orthogonal Matching Pursuit


class OMP:
    """
    Orthogonal Matching Pursuit algorithm for sparse representation.

    :param X: signal to be represented.
    :type X: numpy.ndarray
    :param D: dictionary matrix with atoms as columns
    :type D: numpy.ndarray
    :param MAX_ITER: maximum number of iterations
    :type MAX_ITER: int
    :param EPS: threshold for convergence.
    :type EPS: float

    :return: sparse representation of the signal using the dictionary
    :rtype: numpy.ndarray
    """

    def fit(self, X, D, MAX_ITER, EPS):
        row, col = D.shape
        R = X.copy()

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
            nid_alpha = np.linalg.pinv(Gamma.T @ Gamma) @ Gamma.T @ X

            # indexing alpha
            for i, j in zip(alpha_dict.keys(), range(0, len(nid_alpha))):
                id_alpha[i] = nid_alpha[j]

            # updating R
            R = X - Gamma @ nid_alpha
            k += 1
        return id_alpha
