import numpy as np
import pandas as pd


#df = pd.read_csv('heightShoesSize.csv')
# print(df.describe())
#x = df.to_numpy()
#print(x)


def load_data_linear_regression():
    """
    :return: x values array, y values array.
    """
    df = pd.read_csv('heightShoesSize.csv')
    x = df['Height'].to_numpy()
    y = df['ShoesSize'].to_numpy()
    print(x, y)

    return x, y


def get_cost(x, y, slope, intercept):
    """
    For each x value, calculate the expected y value (Y^) using the slope and the intercept.
    Using the expected y (Y^) and the real y, get the contribution of each data point to the cost.
    Return the overall average cost.

    :param x: array of x values
    :param y: array of (real) y values
    :param slope: slope to use for Y^
    :param intercept: intercept to use for Y^
    :return: average cost.
    """
    y_hat = slope * x + intercept
    sum_cost = sum(y_hat - y) ** 2
    cost = sum_cost / len(x)
    return cost








def get_cost_derivative_wrt_slope(n, x, y_minus_y_hat):
    # YourCodeHere
    rv =  (-2/n) * sum(x*y_minus_y_hat) # Derivative wrt m
    return rv


def get_cost_derivative_wrt_intercept(n, y_minus_y_hat):
    # YourCodeHere
    rv = (-2/n) * sum(y_minus_y_hat)  # Derivative wrt c
    return rv


def get_slope_intercept_gradient_descent(x, y):
    # Building the model
    slope = 0
    intercept = 0

    alpha = 1e-6   # The learning Rate
    epochs = 250000  # The number of iterations to perform gradient descent

    n = float(len(x))  # Number of elements in X

    # Performing Gradient Descent
    for i in range(epochs + 1):
        # Forward propagation
        y_hat = get_expected_ys(x, slope, intercept)  # The current predicted value of Y

        # Backward propagation
        y_minus_y_hat = y-y_hat
        d_slope = get_cost_derivative_wrt_slope(n, x, y_minus_y_hat)
        d_intercept = get_cost_derivative_wrt_intercept(n, y_minus_y_hat)

        # Update parameters
        slope = slope - alpha * d_slope  # Update slope
        intercept = intercept - alpha * 10000 * d_intercept  # Update intercept

        if i % 10000 == 0 or i < 50:
            cost = get_cost(x, y, slope, intercept)
            msg = (f"After {i} epochs: slope= {round(slope, 5)} ,intercept= {round(intercept, 5)}"
                   f" ,cost={round(cost, 5)} ,d_slope={round(d_slope, 5)} ,d_intercept={round(d_intercept,5)}")

            print(msg)

    return slope, intercept


def get_expected_ys(x, slope, intercept):
    """
    Calculate an array of expected y values using our linear formula for x, the slope and the intercept
    :param x: x values to use
    :param slope: slope
    :param intercept: intercept
    :return: array of Y^ i.e. the expected value for each x
    """
    y_hat = x * slope + intercept
    return y_hat


def main():
    print("Bina2.LR Student name: XXX  ,Student ID XXX, my favorite color is green")
    x, y = load_data_linear_regression()

    # Get slope, intercept analytically
    slope, intercept = get_slope_intercept_gradient_descent(x, y)
    print(f"slope= {slope}, intercept= {intercept}")

    # Calculate Cost.
    cost = get_cost(x, y, slope, intercept)
    print(f"Cost= {cost}")


main()