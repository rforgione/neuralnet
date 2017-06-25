from unittest import TestCase
import numpy as np
import math
from network import Network, CrossEntropy

class TestNetwork(TestCase):

    @staticmethod
    def sigmoid(x):
        return np.vectorize(lambda g: 1/(1 + math.exp(-g)))(x)

    @staticmethod
    def sigmoid_to_binary(x):
        return np.vectorize(lambda g: 1 if g >= .5 else 0)(x)

    def test_find_coefficients(self):
        x_matrix = np.random.randn(100, 2)
        coef = np.array([3, 2])
        y_vector = self.sigmoid_to_binary(self.sigmoid(coef.dot(x_matrix.transpose())))

        # input: [x1,
        #         x2]
        #
        # weights: [[ b11, b12 ]
        #

        net = Network([2, 1], CrossEntropy)

        net.train(x_matrix, y_vector, eta=0.5, epochs=80, quiet=True)

        print coef[0]
        print net.weights[0][0]
        assert np.allclose(coef, net.weights[0][0], atol=.5)
