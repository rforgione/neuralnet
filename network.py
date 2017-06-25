import numpy as np
import math
from abc import ABCMeta
from sklearn.metrics import roc_auc_score


class Network(object):

    def __init__(self, architecture, cost):
        self.architecture = architecture
        self.cost = cost
        self.weights = []
        self.biases = []
        self.initialize_weights()
        self.initialize_biases()

    def initialize_weights(self):
        self.weights = [np.random.randn(y, x) for x, y in zip(self.architecture[:-1],
                                                              self.architecture[1:])]

    def initialize_biases(self):
        self.biases = [np.random.randn(x, 1) for x in self.architecture[1:]]

    def feedforward(self, x, full_output=False):
        activations = [x]
        z = []

        for i in range(len(self.weights)):
            z.append(np.dot(self.weights[i], activations[-1]) + self.biases[i])
            activations.append(self.sigmoid(z[-1]))

        if full_output:
            # return a tuple of output values
            return activations, z
        else:
            # return only the last layer output values
            return activations[-1]

    def backprop(self, x, y, cost):
        # there's a single bias for every neuron,
        # so we can uses the biases to double as
        # a representation of the network architecture.
        deltas = [np.zeros(i.shape) for i in self.biases]
        weight_grads = [np.zeros(i.shape) for i in self.weights]
        bias_grads = [np.zeros(i.shape) for i in self.biases]

        activations, z = self.feedforward(x, full_output=True)

        deltas[-1] = cost.delta(activations[-1], y)

        # 2 x 1 -> 1 x 2
        # output x 1 * 1 x 2 = output * activation_length
        weight_grads[-1] = deltas[-1].dot(activations[-2].T)
        bias_grads[-1] = deltas[-1].dot(np.ones(deltas[-1].shape).T)

        for i in xrange(2, len(self.weights)):
            deltas[-i] = np.multiply(np.dot(self.weights[-i + 1], deltas[-i + 1]), z[-i])
            weight_grads[-i] = np.dot(deltas[-i], activations[-i - 1])
            bias_grads[-i] = deltas[-i]

        return weight_grads, bias_grads

    def update_mini_batch(self, mini_batch_x, mini_batch_y, eta=.01):
        n = mini_batch_x.shape[1]

        weight_gradient, bias_gradient = self.backprop(mini_batch_x, mini_batch_y, self.cost)

        weight_adjustment = [(eta/n)*val for val in weight_gradient]
        bias_adjustment = [(eta/n)*val for val in bias_gradient]

        self.weights = [a - b for a, b in zip(self.weights, weight_adjustment)]
        self.biases = [a - b for a, b in zip(self.biases, bias_adjustment)]

    def train(self, x, y, epochs=1, batch_size=None, eta=.01, quiet=False):
        if not batch_size:
            batch_size = x.shape[0]

        for i in xrange(epochs):
            if not quiet:
                print "Beginning epoch %s." % i

            lower_bound = 0
            upper_bound = batch_size
            training_complete = False

            while not training_complete:
                mini_batch_x = x[lower_bound:upper_bound, :].T
                mini_batch_y = y[lower_bound:upper_bound]

                self.update_mini_batch(mini_batch_x, mini_batch_y, eta=eta)

                lower_bound = upper_bound
                upper_bound = lower_bound + batch_size

                if lower_bound >= x.shape[0]:
                    training_complete = True

        return self.get_auc(x, y)

    def get_auc(self, x, y):
        predicted = self.feedforward(x.T)
        return roc_auc_score(y, predicted.ravel(0))


    @staticmethod
    def sigmoid(x):
        def general_func(g):
            return 1/(1+math.exp(-g))

        if isinstance(x, np.ndarray):
            vfunc = np.vectorize(general_func)
            return vfunc(x)
        else:
            return general_func(x)


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

