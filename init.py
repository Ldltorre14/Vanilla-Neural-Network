from keras.datasets import boston_housing
from neuralnetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt



hidden_layer_units_num = 128
output_layer_units_num = 1



#TRAINING DATA
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(f"Training Data Shape: {x_train.shape}")
print(f"Training data: {x_train[:2]}\n")

print(f"Training Targets Shape: {y_train.shape}")
print(f"Training Targets: {y_train[:2]}\n")

print(f"Test Data Shape: {x_test.shape}")
print(f"Test Data: {x_test[:2]}\n")

print(f"Test Targets Shape: {x_train.shape}")
print(f"Test Targets: {y_test[:2]}\n")


#PARAMETERS
weight1 = np.random.randn(x_train.shape[1], hidden_layer_units_num)
weight2 = np.random.randn(hidden_layer_units_num, output_layer_units_num)
biases = None
learning_rate = 0.0001


print(f"Weights Shape: {weight1.shape}")
print(f"Biases Shape: {biases.shape}")


#plt.scatter(x_train[:,1],y_train)
#plt.show()