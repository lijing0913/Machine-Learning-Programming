# Logistic Regression

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def cost_fcn(x, theta, y):
    h = sigmoid(np.dot(x, theta))
    J = np.sum(-y * np.log(h) - (1.0 - y) * np.log(1.0 - h))
    return J

def gradients(x, theta, y):
    h = sigmoid(np.dot(x, theta))
    grads = np.dot(x.T, (h - y))
    return grads
    
def logistic_regression(X, y):
    max_it = 10 # maximum iterations
    alpha = 0.1 # learning rate
    cost = []
    theta = np.random.randn(X.shape[1]) # initial values; dimension of column
    for it in range(max_it):
        cost.append(cost_fcn(X, theta, y))
        grads = gradients(X, theta, y)
        theta = theta - alpha * grads # gradient descend
    pred = np.dot(X, theta)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return pred

x1 = np.random.randn(5,2) + 5
x2 = np.random.randn(5,2) - 5
X = np.concatenate([x1, x2], axis=0)
y = np.concatenate([np.ones(5), -np.ones(5)], axis=0).astype(np.int16)
pred = logistic_regression(X, y)
print(pred) # [1. 1. 1. 1. 1. 0. 0. 0. 0. 0.]]
