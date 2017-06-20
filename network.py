import numpy as np
import math
from abc import ABCMeta

class Network(object):

    def __init__(self, architecture, cost):
        self.architecture = architecture
        self.cost = cost
        self.initialize_weights()
        self.weights = []
        self.biases = []

    def initialize_weights(self):
        self.weights = [np.random.randn([y, x]) for x, y in zip(self.architecture[:-1],
                                                                self.architecture[1:])]

    def initialize_biases(self):
        self.biases = [np.random.randn([x, y]) for x in self.architecture]

    def feedforward(self, x, full_output=False):
        activations = []
        z = [x]
        activations.append(self.sigmoid(z[0]))
        for i in range(len(self.architecture)):
            z.append(np.dot(self.weights[i], activations[-1]) + self.biases[i])
            activations.append(self.sigmoid(z[-1]))

        if full_output:
            # return a tuple of output values
            return activations, z
        else:
            # return only the last layer output values
            return activations[-1]

    def backpropagate(self, X, y, cost):
        # there's a single bias for every neuron,
        # so we can uses the biases to double as
        # a representation of the network architecture.
        deltas = [np.zeros(i.shape) for i in self.biases]
        weight_grads = [np.zeros(i.shape) for i in self.weights]
        bias_grads = [np.zeros(i.shape) for i in self.biases]

        activations, z = self.feedforward(X, full_output=True)

        deltas[-1] = cost.delta(activations[-1], y)
        weight_grads[-1] = np.dot(deltas[-1], activations[-2])
        bias_grads[-1] = deltas[-1]

        for i in xrange(2, len(self.architecture)):
            deltas[-i] = np.multiply(np.dot(self.weights[-i + 1].T, deltas[-i + 1]), z[-i])
            weight_grads[-i] = np.dot(deltas[-i], activations[-i - 1])
            bias_grads[-i] = deltas[-i]
        return weight_grads, bias_grads

    def update_mini_batch(self, mini_batch, eta=.01):
        n = mini_batch.shape[0]
        total_weight_gradient = [np.zeros(self.weights[i].shape) for i in self.weights]
        total_bias_gradient = [np.zeros(self.biases[i].shape) for i in self.biases]

        for j in xrange(n):
            x = mini_batch[j, :-1]
            y = mini_batch[j, -1]
            new_weight_gradient, new_bias_gradient = self.backprop(x, y, self.cost)
            total_weight_gradient = [a + b for a, b in zip(total_weight_gradient, new_bias_gradient)]
            total_bias_gradient = [a + b for a, b in zip(total_bias_gradient, new_bias_gradient)]

        weight_adjustment = [(eta/n)*val for val in self.total_weight_gradient]
        bias_adjustment = [(eta/n)*val for val in self.total_bias_gradient]

        self.weights = [a - b for a, b in zip(self.weights, weight_adjustment)]
        self.biases = [a - b for a, b in zip(self.weights, bias_adjustment)]

    def get_accuracy(self):
        pass

    @staticmethod
    def sigmoid(x):
        return 1/(1+math.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        return Network.sigmoid(x)*(1 - Network.sigmoid(x))

    class Cost(ABCMeta):

        @staticmethod
        def value(x, y):
            pass

        @staticmethod
        def delta(a, y):
            pass

    class CrossEntropy(Cost):

        def __init__(self):
            pass

        @staticmethod
        def value(a, y):
            return -(y*math.log(a) + (1 - y)*math.log(1 - a))

        @staticmethod
        def delta(a, y):
            return a - y

