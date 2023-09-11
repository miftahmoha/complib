import numpy as np

"""
 A class that will treat multidimensional signals.
"""


class Converter:
    def __init__(self, X):
        self.shape = X.shape

    def apply(self):
        return self.X.ravel()

    def reverse(self, Y):
        return Y.reshape(self.shape)
