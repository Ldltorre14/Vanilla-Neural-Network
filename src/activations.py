import numpy as np


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def stepFunctions(z):
    return 1 if z > 0 else 0

def ReLU(z):
    return max(0, z)

def tanh(z):
    g = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return g

def leaky_ReLU(z, alpha=0.01):
    return z if z >= 0 else alpha * z