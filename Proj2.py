import numpy as np
import pandas as pd


def load_data_linear_regression():
    """
    :return: x values array, y values array.
    """
    df = pd.read_csv('NBA_Data_Fixed.csv')
    print(df.describe())
    x1 = df['Height_cm'].to_numpy()
    x2 = df['Weight_pounds'].to_numpy()
    x3 = df['per_suc_field'].to_numpy()
    x4 = df['per_sec_free'].to_numpy()
    y = df['ave_point'].to_numpy()
    print(x1, x2, x3, x4, y)

    return x1, x2, x3, x4, y



def get_cost(x, y, slope, intercept):
    y_hat = slope * x + intercept
    sum_cost = sum(y_hat - y) ** 2
    cost = sum_cost / len(x)
    return cost



def get_cost_derivative_wrt_slope(n, x, y_minus_y_hat):
    rv =  (-2/n) * sum(x*y_minus_y_hat)
    return rv


def get_cost_derivative_wrt_intercept(n, y_minus_y_hat):
    # YourCodeHere
    rv = (-2/n) * sum(y_minus_y_hat)  # Derivative wrt c
    return rv


def get_slope_intercept_gradient_descent(x, y, alpha, epochs, steps):
    # Building the model
    slope = 0
    intercept = 0


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
        mult = 10000
        if steps:
            mult = 100
        intercept = intercept - alpha * mult * d_intercept  # Update intercept

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


def solve_closed_form(x, y):
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x * x)
    sum_xy = np.sum(x * y)
    count = len(x)
    ssr = 0
    sse = 0
    ave_x = sum_x / count
    ave_y = sum_y / count
    slope = (sum_xy - count * ave_x * ave_y) / (sum_x2 - count * ave_x * ave_x)
    intercept = ave_y - slope * ave_x
    for x_i, y_i in zip(x, y):
            y_hat = slope * x_i + intercept
            ssr += (y_hat - ave_y) ** 2
            sse += (y_hat - y_i) ** 2
    sst = ssr + sse
    rx2 = ssr / sst
    print(f"slope= {slope}",f"intercept= {intercept}",f"rx2= {rx2}")
    return slope, intercept, rx2


def main():
    x1, x2, x3, x4, y = load_data_linear_regression()

    get_slope_intercept_gradient_descent(x1, y, 1e-6, 250000, 0)
    solve_closed_form(x1, y)
    get_slope_intercept_gradient_descent(x2, y, 1e-6, 250000, 0)
    solve_closed_form(x2, y)
    get_slope_intercept_gradient_descent(x3, y, 5e-3, 250000, 1)
    solve_closed_form(x3, y)
    get_slope_intercept_gradient_descent(x4, y, 5e-3, 250000, 1)
    solve_closed_form(x4, y)

main()