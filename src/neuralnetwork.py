import numpy as np
from activations import sigmoid,stepFunctions,tanh,ReLU,leaky_ReLU
from lossfn import mse, mae

class NeuralNetwork:
    def __init__(self, x_train, y_train, x_test, y_test,
                 weights, biases, 
                 activation, lossfn, learning_rate,
                 ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.weights = weights
        self.biases = biases
        self.activation = activation
        self.lossfn = lossfn
        self.learning_rate = learning_rate
    
    def buildSequential(self):
        pass
    
        