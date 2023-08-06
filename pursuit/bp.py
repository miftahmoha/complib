import cvxpy as cp
import numpy as np

# Basis Pursuit


class BP:
    def fit(self, X, D):
        """
        Basis Pursuit algorithm for sparse representation.

        :param X: signal to be represented.
        :type X: numpy.ndarray
        :param D: dictionary matrix with atoms as columns
        :type D: numpy.ndarray

        :return: sparse representation of the signal using the dictionary
        :rtype: numpy.ndarray
        """

        row, col = D.shape

        alpha = cp.Variable(col)
        # l1 constraint
        objective = cp.norm(alpha, 1)

        constraints = [D @ alpha == X]
        # solving linear programming problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        return alpha.value
