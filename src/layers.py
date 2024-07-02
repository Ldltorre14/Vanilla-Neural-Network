import numpy as np


def dense(x, weights, biases, activation):    
    """
    Computes dense layer
    Args:
      x          (ndarray (n, )) : Data, 1 sample/vector
      weights    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      biases          (ndarray (j, )) : bias vector, j units  
    Returns
      g          (ndarray (j,))  : j units|
    """
    #Get the unit/neurons size from the second value of the weight's matrix shape
    units = weights.shape[1]
    #Preallocate the memory needed for the output. Just a memory technique for efficiency
    g = np.zeros(units)
    #We iterate through the units/neurons in the layer
    for j in range(units):
        #We get from the weights matrix, all the features from the j columns
        w = weights[:,j]
        #Dot product between the data vector, the weights column and add the current bias
        z = np.dot(x,w) + biases[j]
        #We apply the activation function the z
        g[j] = activation(z)
    return g



def forwardProp(x, w1, w2, b1, b2, activation):
    a1 = dense(x, w1, b1, activation)
    a2 = dense(a1, w2, b2, activation)
    return a2

def backwardProp():
    pass