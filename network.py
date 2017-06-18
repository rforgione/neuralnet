import numpy as np

class Network(object):

    def __init__(self, architecture, cost_function, activation_type):
        self.architecture = architecture
        self.cost_function = cost_function
        self.activation = activation
        self.initialize_weights()
        self.weights = []
        self.biases = []

    def initialize_weights(self):
        self.weights = [np.random.randn([y, x]) for x, y in zip(self.architecture[:-1],
                                                                self.architecture[1:])]

    def initialize_biases(self):
        self.biases = [np.random.randn([x, y]) for x in self.architecture]

    def feedforward(self):
        pass

    def backpropagate(self):
        pass

    def update_mini_batch(self):
        pass

    def get_accuracy(self):
        pass

