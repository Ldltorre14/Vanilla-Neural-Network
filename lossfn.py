import numpy as np


def mse(y_pred, y_true):
    squared_error = (y_pred - y_true) ** 2
    sum_squared_error = np.sum(squared_error)
    loss = sum_squared_error / y_true.size
    return loss

def mae(y_pred, y_true):
    abs_squared_error = abs(y_pred - y_true)
    sum_abs_error = np.sum(abs_squared_error)
    loss = sum_abs_error / y_true.size
    
def huberLoss(y_pred, y_true):
    pass