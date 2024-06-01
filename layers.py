import numpy as np


def dense(units, x, weights, bias, activation):    
    """
    units = neurons in the layer
    x = np.Array(m,n)   : float
    weights = np.Array(n,) : float
    biases = number : float
    activation = activation function 
    """
    for i in range(units):
        z = np.dot(x[i], weights) + bias
        g = activation(z)
    return g

    
    
    
        
    

def forwardProp(x, w1, w2, b1, b2, activation):
    pass