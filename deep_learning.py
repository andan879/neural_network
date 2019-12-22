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

if True:
	nn = NeuralNetwork([2, 3, 1])
	for lay_nr, bias in enumerate(nn.biases):
		print('biases in layer ', lay_nr + 2, ': \n', bias)

	for lay_nr, weight in enumerate(nn.weights):
		print('weights in layer ', lay_nr + 2, ': \n', weight)
