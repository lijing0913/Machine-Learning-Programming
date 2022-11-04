# Activation functions
import numpy as np
import matplotlib.pyplot as plt

# define the activation functions
def relu(z):
	return np.maximum(0, z)

def sigmoid(z):
	return 1. / (1. + np.exp(-z))

def softplus(z):
	return np.log(1 + np.exp(z))

def softmax(z):
	return np.exp(z) / np.sum(np.exp(z), axis=0)

# Plot the activation functions
def plot_fcn(f, ax):
	x = np.arange(-5, 5 + 0.1, 0.1)
	ax.plot(x, f(x))
	title = f.__name__
	ax.set_title('%s' % title)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
plot_fcn(relu, axes[0,0])
plot_fcn(sigmoid, axes[0,1])
plot_fcn(softplus, axes[1,0])
plot_fcn(softmax, axes[1,1])
plt.tight_layout()
plt.show()