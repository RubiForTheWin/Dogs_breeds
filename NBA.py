import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

df= pd.read_csv('NBA_Data_Fixed.csv')
#print(df.describe())
def load_data_linear_regression():
    w = df['Weight_pounds'].to_numpy()
    h = df['Height_cm'].to_numpy()
    field = df['per_suc_field'].to_numpy()
    free = df['per_sec_free'].to_numpy()
    y = df['ave_point'].to_numpy()


    return w,h,field,free,y

def solve_sklearn(x, y):
    print("solve_sklearn:")
    reg = LinearRegression()
    y=y.reshape(-1,1)
    x=x.reshape((-1,1))
    reg.fit(x, y)
    print(f"coefficients= {reg.coef_}")
    print(f"intercept= {reg.intercept_}")
    print(f"score= {reg.score(x, y)}")

    y_hat = reg.predict(x)
    count = len(y)
    cost = np.sum((y_hat - y) ** 2) / count
    print(f"cost= {cost}")

def get_slope_intercept(x, y):
    sum_x2 = np.sum(x * x)
    sum_xy = np.sum(x * y)

    ave_x = np.average(x)
    ave_y = np.average(y)
    count = len(x)

    slope = (sum_xy - count * ave_x * ave_y) / (sum_x2 - count * ave_x * ave_x)
    intercept = ave_y - slope * ave_x
    return slope, intercept

def solve_closed_form(x,y):
    slope, intercept=get_slope_intercept(x, y)
    ave_y = np.average(y)
    y_hat = x * slope + intercept
    ssr = np.sum((y_hat - ave_y) ** 2)
    sst = np.sum((y - ave_y) ** 2)
    r2 = ssr/sst

    print(slope, intercept, r2)
    return r2
def get_cost(x, y, slope, intercept):
    sum_cost = 0
    y_hat = slope * x + intercept
    sum_cost = np.sum((y_hat - y) ** 2)
    cost = sum_cost / len(x)
    return cost


def get_cost_derivative_wrt_slope(n, x, y_minus_y_hat):
    # YourCodeHere

    rv =(-2/n)* sum(x * y_minus_y_hat)
    return rv


def get_cost_derivative_wrt_intercept(n, y_minus_y_hat):
    # YourCodeHere
    rv = (-2/n)*sum(y_minus_y_hat)
    return rv

def solve_gradient_descent(x, y, alpha, epochs, big_steps):
    # Building the model
    slope = 0
    intercept = 0

    n = float(len(x))  # Number of elements in X

    # Performing Gradient Descent
    for i in range(epochs + 1):
        # Forward propagation
        y_hat = x * slope + intercept  # The current predicted value of Y

        # Backward propagation
        y_minus_y_hat =y - y_hat
        d_slope = get_cost_derivative_wrt_slope(n, x, y_minus_y_hat)
        d_intercept = get_cost_derivative_wrt_intercept(n, y_minus_y_hat)

        # Update parameters
        slope = slope - alpha * d_slope  # Update slope
        mult = 10000
        if big_steps:
            mult = 100
        intercept = intercept - alpha * mult * d_intercept  # Update intercept

        if i % 10000 == 0 or i < 50:
            cost = get_cost(x, y, slope, intercept)
            msg = (f"After {i} epochs: slope= {round(slope, 5)} ,intercept= {round(intercept, 5)}"
                   f" ,cost={round(cost, 5)} ,d_slope={round(d_slope, 5)} ,d_intercept={round(d_intercept,5)}")

            print(msg)

    return slope, intercept

def main():
    w,h,field,free,y  = load_data_linear_regression()

    r_squared=solve_closed_form(w, y)
    solve_gradient_descent(w, y, 1e-6, 250000, 0)
    print(f'rx2 for weight {r_squared}')
    slope, intercept = get_slope_intercept(w, y)
    y_hat = w * slope + intercept
    solve_sklearn(w,y)
    plt.plot(w, y, 'bo', w, y_hat, 'r-')
    plt.suptitle("average point vs. height")
    plt.xlabel("height")
    plt.ylabel("average points")
    plt.show()
    r_squared =  solve_closed_form(h, y)
    solve_gradient_descent(h, y, 1e-6, 250000,0)
    print(f'rx2 for height {r_squared}')
    slope, intercept = get_slope_intercept(h, y)
    y_hat = h* slope + intercept
    plt.plot(h, y, 'bo', h, y_hat, 'r-')
    plt.suptitle("average point vs. height")
    plt.xlabel("height")
    plt.ylabel("average points")
    plt.show()
    r_squared = solve_closed_form(field, y)
    solve_gradient_descent(field, y, 5e-3, 250000, 1)
    print(f'rx2 for per_sec_feild {r_squared}')
    slope, intercept = get_slope_intercept(field, y)
    y_hat = field * slope + intercept
    plt.plot(field, y, 'bo', field, y_hat, 'r-')
    plt.suptitle("average point vs. height")
    plt.xlabel("height")
    plt.ylabel("average points")
    plt.show()
    r_squared = solve_closed_form(free, y)
    solve_gradient_descent(free, y, 5e-3 ,250000, 1)
    print(f'rx2 for per_sec_free {r_squared}')
    slope, intercept = get_slope_intercept(free, y)
    y_hat = free * slope + intercept
    plt.plot(free, y, 'bo', free, y_hat, 'r-')
    plt.suptitle("average point vs. height")
    plt.xlabel("height")
    plt.ylabel("average points")
    plt.show()
    # Calculate Cost.


if  __name__ == '__main__':
    main()