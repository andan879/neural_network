# Standard libraries
import random

# Third-pary libraries
import numpy as np

class NeuralNetwork(object):
	'''
	sizes is and array, where each element
	represent the number of neurons in each
	layer. 

	Eg:
	net = NeuralNetwork([2, 3, 1])
	'''
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes


		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

		self.weights = [np.random.randn(y, x)
						for x, y in zip(sizes[:-1], sizes[1:])]

		def sigmoid(self, z):
			return 1 / (1 + np.exp(-z))

		def feed_forward(self, a):
			for b, w in zip(self.biases, self.weights):
				a = sigmoid(np.dot(w, a) + b)
			return a

# Log some NN properties
if True:
	divider = '------------------------------------------------------'
	nn = NeuralNetwork([2, 3, 1])

	print('###################### Neural Network - Pretty Print ############################')
	print('Layer design: \n' + str(nn.sizes))

	print(divider)

	for i, bias in enumerate(nn.biases):
		print"Bias(es) in layer nr {0}: \n".format(i+2), str(bias)

	print(divider)
 
	for i, weight in enumerate(nn.weights):
		print "Weight(s) connecting layer {0} with layer {1}: \n".format(i, i+1), str(weight)

	print(divider)

	print zip(nn.biases, nn.weights)

