# reference: neural network - learning from data - Yaser Abu Mostafa
# http://work.caltech.edu/slides/slides10.pdf
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    return 1 - tanh(x) * tanh(x)

class NeuralNetwork:
    def __init__(self, sizes):
        # Sizes: sizes of neural net
        # Ex: [2, 3, 1]: it has 3 layers, input layer has 2 features
        self.sizes = sizes
        self.biases = [] # biases[i]: biases of layer i + 1
        self.x = [] # output of layer i + 1
        self.signals = [] # signal of layer i + 1
        self.deltas = []

        for i in range(1, len(sizes)):
            self.biases.append(np.random.randn(sizes[i], 1))
            self.x.append(np.zeros((sizes[i], 1)))
            self.signals.append(np.zeros((sizes[i], 1)))
            self.deltas.append(np.zeros((sizes[i], 1)))

        # Weight (Weights[i][a][b]: weights of element [b] of layer i to element [a] of layer[i + 1])
        self.weights = []
        for i in range(len(sizes) - 1):
            self.weights.append(np.random.randn(sizes[i + 1], sizes[i]))

    def stochastic_gradient_descent(self, train_data, test_data=None, epochs=100, learning_rate=0.1):
        train_x, train_y = train_data
        for epoch in range(epochs):
            index = np.random.permutation(train_x.shape[0])

            # sgd
            for i in index:
                self.feed_forward(train_x[i])
                

    def feed_forward(self, inputs):
        self.signals[0] = np.dot(self.weights[0], inputs) + self.biases[0]
        self.x[0] = sigmoid(self.signals[0])
        for i in range(1, len(self.weights)):
            self.signals[i] = np.dot(self.weights[i], self.x[i - 1]) + self.biases[i]
            self.x[i] = sigmoid(self.signals[i])



if __name__ == "__main__":
    pass
