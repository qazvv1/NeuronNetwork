import numpy as np

def sigmoid(x):
	#function f(x) = 1 / (1+e^(-x))
	return 1 / (1 + np.exp(-x))

class Neuron():
	"""docstring for Neuron"""
	def __init__(self, weight, bias):
		self.weight = weight
		self.bias = bias

	def feedforward(self, input):
		total = np.dot(self.weight, input) + self.bias
		return sigmoid(total)


class NeuronNetwork():
	"""docstring for NeuonNetwork"""
	def __init__(self):
		self.h1 = Neuron(np.array([1, 1]), 0)
		self.h2 = Neuron(np.array([1, 1]), 0)
		self.o1 = Neuron(np.array([1, 1]), 0)

	def feedforward(self, input):
		out_h1 = self.h1.feedforward(input)
		out_h2 = self.h2.feedforward(input)
		out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
		return out_o1

#####################################################

no = NeuronNetwork()
x = np.array([-2, -1])
out = no.feedforward(x)

print(out)