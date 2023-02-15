import pandas as pd
import numpy as np
from Regressor import Regressor


def load_data():
    data_frame = pd.read_csv("HoursOfStudy10K.csv")
    print(data_frame.describe())
    print("-------------------")

    print(data_frame.head())
    print("-------------------")

    # Get X, y
    y = data_frame.iloc[:, -1].to_numpy().reshape(1, -1)
    x = data_frame.iloc[:, 0:-1].to_numpy().reshape(1, -1)

    return x, y


def sigmoid(z):
    rv = 1/(1 + np.exp(-z))
    return rv


def cross_entropy(y_hat, y):
    rv = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return rv


def main():
    # Get the data
    x_training_data, y_training_data = load_data()

    # Create the model
    model = Regressor(sigmoid, cross_entropy)
    model.fit(x_training_data, y_training_data)
    x_test_data = np.array([[1.6, 1.3]])
    predictions = model.predict(x_test_data) >= 0.5
    print(x_test_data, predictions.T)


if __name__ == '__main__':
    main()
