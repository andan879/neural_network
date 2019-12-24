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

		def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
			'''
			Stochastic Gradient Descent.
			--------------------------------------------------------
			training_data: 
				list of tuples (x, y) where x is the training input
				and y is the corresponding desired output
			--------------------------------------------------------
			epochs:
				the number of epochs to train for
			--------------------------------------------------------
			mini_batch_size:
				size of the mini_batch when sampling
			--------------------------------------------------------
			eta:
				the learning rate
			--------------------------------------------------------
			test_data:
				if test_data is supplied, the program will evaluate
				the network after each epoch and print the partial
				progress
			--------------------------------------------------------

			'''
			if test_data:
				n_test = len(test_data)
			for j in range(epochs):
				random.shuffle(training_data)
				mini_batches = [training_data[k:k + mini_batch_size] for k in (0, n, mini_batch_size)]
				 
				for mini_batch in mini_batches:
					self.update_mini_batch(mini_batch, eta)

				if test_data:
					print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
				else:
					print "Epoch {0} complete".format(j)

		def update_mini_batch(self, mini_batch, eta):
			'''
			Updates the weights and biases according to a single
			iteration of SGD. 
			--------------------------------------------------------
			mini_batch:
				a mini_batch is a list of tuples, (x, y)
			--------------------------------------------------------
			eta:
				the learning rate
			--------------------------------------------------------  
			'''
			nabla_b = [np.zeros(b.shape) for b in self.biases]
			nable_w = [np.zeros(w.shape) for w in self.weights]

			for x, y in mini_batch:
				delta_nabla_b, delta_nabla_w = self.backdrop(x, y)
				nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
				nabla_w = [nw + dnw for nw, dnw in zip(nable_w, delta_nabla_w)]

			self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]

			self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

		def backdrop(x, y):
			pass








# Log some NN properties
if True:
	divider = '------------------------------------------------------'
	nn = NeuralNetwork([2, 3, 1])

	print('###################### Neural Network - Pretty Print ############################')
	print('Layer design: \n' + str(nn.sizes))

	print(divider)


