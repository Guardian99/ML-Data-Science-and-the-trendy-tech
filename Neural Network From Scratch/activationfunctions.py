import numpy as np

# ==================================================
# Forward Propagation
def Sigmoid(z):
	sigmoidtransform=1.0 / (1.0 + np.exp(-z))
	return sigmoidtransform

def ReLu(z):
	relutransform=np.maximum(z, 0)
	return relutransform

def Linear(z):
	return z

def Tanh(z):
	tanhtransform=np.tanh(z)
	return tanhtransform
# ==================================================
# Anti Derivatives
def Sigmoid_prime(z):
	return Sigmoid(z) * (1 - Sigmoid(z))

def Linear_prime(z):
	return np.ones((z.shape))

def ReLu_prime(z):
	return z > 0

def Tanh_prime(z):
	return (1 - (Tanh(z) * Tanh(z)))
# ==================================================
# For Output Layer
def softmax(z):
	return np.exp(z) / np.sum(np.exp(z))
