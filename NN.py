import numpy as np
import sys


def sigmoid(z):
    z = 1 / (1 + np.exp(-z))
    return z


def dSigmoid_dZ(A,  dA):
    """ sigmoid function derivation """
    return A * (1 - A) * dA


def relu(Z):
    """ relu function """
    return np.maximum(0, Z)


def dRelu_dZ(A, dA):
    """ relu function derivation """
    dZ = np.where(A <= 0, 0, 1) * dA
    return dZ


def tanh(Z):
    """ tanh function """
    return np.tanh(Z)


def dTanh_dZ(A, dA):
    """ tanh function derivation """
    dTan = 1 - A * A
    return dTan * dA


def softmax(Z):
    """ softmax method """
    e = np.exp(np.clip(Z, None, 600))
    rv = e / np.sum(e, axis=0)
    return rv


def softmax_dZ(Z, dZ):
    """
    softmax gradient
    actually does nothing - calculation done on previous layer
    """
    return dZ


def square_dist(Y_hat, Y):
    """ square distance calculation """
    errors = (Y_hat - Y)**2
    return errors


def dSquare_dist(Y_hat, Y):
    """ square distance formula derivation """
    m = Y.shape[1]
    dY_hat = 2*(Y_hat - Y)/m
    return dY_hat


def cross_entropy(Y_hat, Y):
    """ cross entropy function calculation """
    return -(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))


def dCross_entropy(Y_hat, Y):
    """ cross entropy function gradient """
    return (-(Y / Y_hat) + ((1 - Y) / (1 - Y_hat)))/Y.shape[1]


def categorical_cross_entropy(Y_hat, Y):
    """ categorical cross entrropy method """
    tmp = Y * np.log(np.clip(Y_hat, 1e-300, None))
    return -np.sum(tmp, axis=0)


def dCategorical_cross_entropy(Y_hat, Y):
    """
    categorical_cross_entropy gradient
    here it is already multiplies by next layer gradient
    """
    return Y_hat - Y


class Layer:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size

        # Works for MNIST
        self.w = np.random.randn(output_size, input_size) * 0.01

        # Works for XoR
        self.w = np.random.rand(output_size, input_size).round(decimals=1)

        self.b = np.zeros((output_size, 1))
        self.z = None
        self.a = None
        self.x = None
        self.dz = None
        self.dw = None
        self.db = None

        if activation == "sigmoid":
            self.activation_function = sigmoid
            self.d_activation_function = dSigmoid_dZ
        elif activation == "relu":
            self.activation_function = relu
            self.d_activation_function = dRelu_dZ
        elif activation == "tanh":
            self.activation_function = tanh
            self.d_activation_function = dTanh_dZ
        elif activation == "softmax":
            self.activation_function = softmax
            self.d_activation_function = softmax_dZ
        else:
            print(f"Do not know what to do with activation function {activation}, quitting.")
            sys.exit(-1)

    def forward_propagation(self, x):
        self.x = x
        self.z = np.dot(self.w, x) + self.b
        self.a = self.activation_function(self.z)

    def back_propagation(self, last_output, last_weights, m):
        if last_weights is None:
            self.dz = self.a - last_output
        else:
            # self.dz = np.dot(last_weights.T, last_output) * self.a * (1 - self.a)
            self.dz = self.d_activation_function(self.a, np.dot(last_weights.T, last_output))
        self.dw = np.dot(self.dz, self.x.T) / m
        self.db = np.sum(self.dz, axis=1, keepdims=True) / m

    def update(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db


class NN:
    def __init__(self, loss, learning_rate, epochs=10000, verbose=0):
        self.layers = []
        self.learning_rate = learning_rate
        self.iterations = epochs
        self.losses = []
        self.verbose = verbose
        if loss == "square_dist":
            self.loss_forward = square_dist
            self.loss_backward = dSquare_dist
        elif loss == "cross_entropy":
            self.loss_forward = cross_entropy
            self.loss_backward = dCross_entropy
        elif loss == "categorical_cross_entropy":
            self.loss_forward = categorical_cross_entropy
            self.loss_backward = dCategorical_cross_entropy
        else:
            raise(Exception(f"Do not know how to handle loss {loss}"))

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, x, y):
        for i in range(self.iterations):
            input_for_next_layer = x
            for layer_index, layer in enumerate(self.layers):
                layer.forward_propagation(input_for_next_layer)
                input_for_next_layer = layer.a

            output = self.layers[-1].a
            m = y.shape[1]

            loss = np.sum(self.loss_forward(output, y)) / m
            if self.verbose and (i < 10 or i % 100 == 0):
                print(f"Iteration {i}, loss= {loss}")
            self.losses.append(loss)

            last_output = y
            last_weights = None
            for layer in reversed(self.layers):
                layer.back_propagation(last_output, last_weights, m)
                last_output = layer.dz
                last_weights = np.copy(layer.w)
                layer.update(self.learning_rate)

        if self.verbose > 1:
            for i, layer in enumerate(self.layers):
                for j in range(len(layer.b)):
                    print(f"Layer {i} w{j}= {layer.w[j]} ,b={layer.b[j]} a={np.round(layer.a[j,:])}")

    def test(self, x):
        input_for_next_layer = x
        for layer in self.layers:
            layer.forward_propagation(input_for_next_layer)
            input_for_next_layer = layer.a

        return input_for_next_layer
