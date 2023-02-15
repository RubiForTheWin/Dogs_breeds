from NN_Xor_Mnist import *

from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Take in an array with classes from 0, 1, ... n, with m number of elements
# Returns a one hot encoded matrix of shape (n, m)
def one_hot_array(Y):
    b = np.zeros((Y.size, Y.max() + 1))
    b[np.arange(Y.size), Y] = 1
    return b.T


def plot_digit(images, digits, index):
    image = images[:, index]
    digit = digits[:, index].argmax()
    im_reshape = image.reshape(28, 28)
    plt.imshow(im_reshape, cmap='Greys')
    plt.title("The label is: " + str(digit))
    plt.show()


def load_data():
    train = pd.read_csv("./Mnist.csv")
    X = train.iloc[:, 1:].values
    Y = train.iloc[:, 0]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    X_train = X_train.T
    X_test = X_test.T
    Y_train = one_hot_array(Y_train.values)
    Y_test = one_hot_array(Y_test.values)

    print(f"{datetime.now()} Shape of X_train is: " + str(X_train.shape))
    print(f"{datetime.now()} Shape of X_test is: " + str(X_test.shape))
    print(f"{datetime.now()} Shape of Y_train is: " + str(Y_train.shape))
    print(f"{datetime.now()} Shape of Y_test is: " + str(Y_test.shape))

    return X_train, Y_train, X_test, Y_test


def main():
    np.random.seed(3)
    print(f"{datetime.now()} Starting {sys.argv[0]}")
    x_train, y_train, x_test, y_test = load_data()
    print(f"{datetime.now()} After load_data()")

    layer1 = Layer(784, 10, "relu")
    layer2 = Layer(10, 10, "softmax")

    nn = NN("categorical_cross_entropy", learning_rate=0.001, epochs=100, verbose=1)
    nn.add_layer(layer1)
    nn.add_layer(layer2)

    # plot_digit(x_train, y_train, YOUR_NUMBER_HERE)

    nn.train(x_train, y_train)

    print(f"Train set accuracy is {nn.get_accuracy(x_train, y_train)}")
    print(f"Test set accuracy is {nn.get_accuracy(x_test, y_test)}")

    nn.confusion_matrix("Test Data Confusion Matrix", x_test, y_test)

    print(f"{datetime.now()} Done")


main()
