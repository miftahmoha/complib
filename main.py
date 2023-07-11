from pursuit import MP, OMP, StOMP, CoSaMP
from learning import ksvd
import numpy as np
import pandas as pd
import time


# Signal & Dictionary
D_EVAL = np.array(
    [
        [np.sqrt(2) / 2, np.sqrt(3) / 3, np.sqrt(6) / 3, 2 / 3, -1 / 3],
        [-np.sqrt(2) / 2, -np.sqrt(3) / 3, -np.sqrt(6) / 6, 2 / 3, -2 / 3],
        [0, -np.sqrt(3) / 3, np.sqrt(6) / 6, 1 / 3, 2 / 3],
    ],
    float,
)
X_EVAL = np.array([4 / 3 - np.sqrt(2) / 2, 4 / 3 + np.sqrt(2) / 2, 2 / 3], float)

# MP

MAX_ITER = 5000
EPS = 1e-2
start_time = time.time()
mp = MP()

alpha = mp.fit(X_EVAL, D_EVAL, MAX_ITER, EPS)

print(f"MP: {alpha}")
end_time = time.time()
execution_time = end_time - start_time
print("Execution time for MP:", execution_time, "seconds")

# OMP

MAX_ITER = 5000
EPS = 1e-2
start_time = time.time()
omp = OMP()

alpha = omp.fit(X_EVAL, D_EVAL, MAX_ITER, EPS)

print(f"MP: {alpha}")
end_time = time.time()
execution_time = end_time - start_time
print("Execution time for OMP:", execution_time, "seconds")

# StOMP

MAX_ITER = 5000
EPS = 1e-2
start_time = time.time()
stomp = StOMP()

alpha = stomp.fit(X_EVAL, D_EVAL, MAX_ITER, EPS, 1.5)

print(f"MP: {alpha}")
end_time = time.time()
execution_time = end_time - start_time
print("Execution time for StOMP:", execution_time, "seconds")

# CoSaMP

MAX_ITER = 5000
EPS = 1e-2
start_time = time.time()
cosamp = CoSaMP()

alpha = cosamp.fit(X_EVAL, D_EVAL, MAX_ITER, EPS, 3)

print(f"MP: {alpha}")
end_time = time.time()
execution_time = end_time - start_time
print("Execution time for StOMP:", execution_time, "seconds")


# Dictionary learning

# K-SVD
X = pd.read_excel("data.xlsx", "data_train", nrows=100)

KSVD = ksvd.KSVD()
# hyperparameters
K = 25  # number of atoms
MAX_ITER = 5000
EPS = 1e-2
solver = OMP()

D = KSVD.fit(X, K, solver, EPS, MAX_ITER)
print("Learned dictionary: ", D)
