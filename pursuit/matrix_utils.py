import numpy as np

"""
 A class that will treat multidimensional signals.
"""


class Converter:
    def __init__(self, X):
        self.X = X


"""
Dictionaries

"""


# Identity Dictionary
def identity_dictionary(n):
    return np.eye(n)


# Haar Dictionary
def haar_dictionary(n):
    # Generate a dictionary of 1D Haar wavelets of dimension nxn
    haar_dict = []
    for i in range(n):
        # Generate 1D Haar wavelet at position i
        haar_wavelet = np.zeros(n)
        haar_wavelet[i : i + 2] = 1
        haar_dict.append(haar_wavelet)
    return np.array(haar_dict)


# Discrete Cosine Transform Dictionary
def dct_dictionary(size, num_atoms):
    dct_dict = np.zeros((size, num_atoms))
    for k in range(num_atoms):
        basis = np.cos(np.arange(size) * k * np.pi / num_atoms)
        if k > 0:
            basis -= np.mean(basis)
        dct_dict[:, k] = basis / np.linalg.norm(basis)
    return dct_dict


"""
Measure matrices

"""


#  Random Gaussian Matrix
def gaussian_matrix(rows, cols):
    return np.random.randn(rows, cols)


# Random Bernoulli Matrix
def bernoulli_matrix(rows, cols):
    return np.random.choice([-1, 1], size=(rows, cols))
