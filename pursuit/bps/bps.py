"""
Multitask Compressive Sensing: Shihao Ji, David Dunson†, and Lawrence Carin

https://www.researchgate.net/publication/224514217_Multitask_Compressive_Sensing
"""

import stan
import numpy as np

import importlib
import os
import sys
from copy import deepcopy

# adding the parent directory to the sys.path list to import utils module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


# checking that input is a numpy array
def check_numpy_array(input_data):
    if not isinstance(input_data, np.ndarray):
        raise TypeError("Input must be a NumPy array.")


# normalizing measure matrix
def normalize_columns(matrix):
    col_norms = np.linalg.norm(matrix, axis=0)
    normalized_matrix = matrix / col_norms
    return normalized_matrix


# reading the BPS stan model
with open("./pursuit/bps/bps.stan", "r") as file:
    bps_stan = file.read()


def process_results(results, M, n):
    def create_theta_dictionary(M, n):
        theta_dict = {}
        for i in range(1, M + 1):
            key = f"Theta.{i}"
            theta_dict[key] = np.zeros(n)
        return theta_dict

    df = results.to_frame().describe().T
    theta_dict = create_theta_dictionary(M, n)
    for i in range(1, n * M + 1):
        key_dict = f"Theta.{(i - 1) % M + 1}"
        key_df = f"Theta.{(i - 1) % M + 1}.{(i - 1) % n + 1}"
        theta_dict[key_dict][(i - 1) % n] = df.at[key_df, "mean"]
    return theta_dict


class BPS:
    def fit(self, X, dictionary="dct", measure_matrix="gaussian", proportion=0.5):
        # X must be a numpy array
        check_numpy_array(X)

        # importing utils module
        try:
            imported_module = importlib.import_module("matrix_utils")
            print(f"Successfully imported utils module.")

        except ModuleNotFoundError:
            print(
                f"Module '{dictionary}' not found. Please check the module name and try again."
            )

        # importing the dictionary
        try:
            generate_dict = getattr(imported_module, f"{dictionary}_dictionary")

        except NameError:
            print(f"{dictionary} not found. Please check the name and try again.")

        # importing the measure matrix
        try:
            generate_measure = getattr(imported_module, f"{measure_matrix}_matrix")
        except NameError:
            print(f"{measure_matrix} not found. Please check the name and try again.")

        # original signals
        Z = deepcopy(X).T
        Z_centered = Z - np.mean(Z, axis=1, keepdims=True)
        Z_norms = np.linalg.norm(Z_centered, axis=1, keepdims=True)
        Z = Z_centered / Z_norms

        # M: number of signals, n: original dimension of signals
        M, n = Z.shape

        # measure dimension
        m = int(n * proportion)

        # measure matrix
        Phi = normalize_columns(generate_measure(m, n))

        # measures vectors
        Z_measure = Z @ Phi.T

        # sampling from STAN model
        signals_data = {"M": M, "n": n, "m": m, "X": Z_measure, "phi": Phi}
        posterior = stan.build(bps_stan, data=signals_data)
        fit_results = posterior.sample(num_chains=4, num_samples=1)
        results = process_results(fit_results, M, n)
        return results
