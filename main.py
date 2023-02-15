import matplotlib.pyplot as plt


def load_data_linear_regression():
    """
    :return: x values array, y values array.
    """
    x = [163, 184, 170, 160, 172, 183, 183, 175, 177, 180, 184, 170, 170, 160, 158, 167, 166, 175, 180, 174, 171]
    y = [39, 45, 43, 37, 45, 43, 45, 43, 43, 44, 44, 42, 42, 38, 37, 40, 38, 44, 45, 43, 44]

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
    sum_cost = 0
    for x_i, y_i in zip(x, y):
        pass
        # YourCodeHere5: use x_i, intercept and slope in order to calculate y_hat

        # YourCodeHere6: use y_hat and y_i in order to get the contribution of this data point to the cost (cost_i)
        # YourCodeHere7: aggregate cost_i into sum_cost

    cost = sum_cost / len(x)
    return cost


def plot_cost_vs_intercept(x, y, slope, intercept):
    array_intercepts = []
    array_costs = []
    for i in range(-50, 50):
        intercept_i = intercept + i / 100
        cost = get_cost(x, y, slope, intercept_i)
        # print(i, intercept_i, cost)
        array_intercepts.append(intercept_i)
        array_costs.append(cost)
    plt.plot(array_intercepts, array_costs, 'o', color='black')
    plt.suptitle("Cost vs. Intercept")
    plt.show()


def plot_cost_vs_slope(x, y, slope, intercept):
    """
    Loop through modified intercept values.
    for each intercept value get the resulting cost.
    store each intercept and its cost in an array.
    plot the array.

    :param x: array of x values to use.
    :param y: array of y values to use.
    :param slope: slope
    :param intercept: intercept
    :return: No return value.
    """
    array_slopes = []
    array_costs = []
    for i in range(-50, 50):
        slope_i = slope + i / 100
        cost = get_cost(x, y, slope_i, intercept)
        # print(i, a_i, cost)
        array_slopes.append(slope_i)
        array_costs.append(cost)
    plt.plot(array_slopes, array_costs, 'o', color='black')
    plt.suptitle("Cost vs. Slope")
    plt.show()


def get_slope_intercept(x, y):
    """
    Calculates slope and intercept for Linear Regression using the formulas from:
    https://gerireshef.wordpress.com/2011/10/31/%D7%A8%D7%92%D7%A8%D7%A1%D7%99%D7%94-%D7%9C%D7%99%D7%A0%D7%90%D7%A8%D7%99%D7%AA-%D7%A9%D7%99%D7%98%D7%AA-%D7%94%D7%A8%D7%99%D7%91%D7%95%D7%A2%D7%99%D7%9D-%D7%94%D7%A4%D7%97%D7%95%D7%AA%D7%99/

    :param x: X values to use
    :param y: Y values to use
    :return: slope, intercept
    """

    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_xy = 0

    # YourCodeHere1: At the end of this lo  op, sum_x, sum_y, sum_x2, sum_xy should have the right values
    for x_i, y_i in zip(x, y):
        sum_x += x_i
        sum_y += y_i
        sum_x2 += (x_i + x_i)
        sum_xy += (x_i + y_i)

    count = len(x)
    ave_x = sum_x / count
    ave_y = sum_y / count

    slope = 0  # YourCodeHere2: You should change the "0" with the right formula from the website above
    intercept = 0  # YourCodeHere3: You should change the "0" with the right formula from the website above

    return slope, intercept


def get_expected_ys(x, slope, intercept):
    """
    Calculate an array of expected y values using our linear formula for x, the slope and the intercept
    :param x: x values to use
    :param slope: slope
    :param intercept: intercept
    :return: array of Y^ i.e. the expected value for each x
    """
    y_hat = []
    for x_i in x:
        y_hat.append(0)  # YourCodeHere4: change the "0" to the right formula

    return y_hat


def main():
    print("Bina1.LR Student name: XXX  ,Student ID XXX")

    x, y = load_data_linear_regression()
    print(x, y)

    # Get slope, intercept analytically
    slope, intercept = get_slope_intercept(x, y)
    print(f"slope= {slope}, intercept= {intercept}")

    y_hats = get_expected_ys(x, slope, intercept)

    # Plot Y vs X and Y_hats vs x
    plt.plot(x, y, 'bo', x, y_hats, 'r-')
    plt.suptitle("Y vs. X")
    plt.show()

    # Calculate Cost.
    cost = get_cost(x, y, slope, intercept)
    print(f"Cost= {cost}")

    plot_cost_vs_slope(x, y, slope, intercept)

    plot_cost_vs_intercept(x, y, slope, intercept)


if _name_ == '_main_':
    main()