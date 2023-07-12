import cvxpy as cp
import numpy as np

# Basis Pursuit


class BP:
    def fit(self, X, D):
        row, col = D.shape
        alpha = cp.Variable(col)
        objective = cp.norm(alpha, 1)
        constraints = [D @ alpha == X]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        return alpha.value
