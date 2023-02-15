import numpy as np
import matplotlib.pyplot as plt


def reshape():
    a = np.array([1, 2, 3, 4, 5, 6])
    print(f"a={a}")
    print(f"a.shape={a.shape}")
    print("________")

    r6c1 = a.reshape(6, 1)

    print(f"r6c1={r6c1}")
    print(f"r6c1.shape={r6c1.shape}")
    print("________")

    r3c2 = a.reshape(3, 2)

    print(f"r3c2={r3c2}")
    print(f"r3c2.shape={r3c2.shape}")
    print("________")

    print(f"r3c2.T={r3c2.T}")
    print(f"r3c2.t.shape={r3c2.T.shape}")
    print("________")

    b = a.reshape(-1, 2)
    print(f"b={b}")
    print(f"b.shape={b.shape}")
    print("________")


def matrix_operation():
    a = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)
    b = np.array([1, 2])
    ab = a@b
    print(f"ab={ab}")


def sigmoid(x):
    rv = 1/(1+np.exp(-x))
    return rv


def plot():
    # הגדרת מערך x
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    z = -x
    print(y)
    plt.figure()
    plt.plot(x, y, 'pink', x, z, 'blue')
    plt.show()


def plot_histogram():
    # הגרלת מספרים רנדומלים
    x = np.random.randn(10000)
    plt.hist(x)
    plt.show()


def main():
    # reshape()

    # matrix_operation()
    # plot()
    plot_histogram()
    print("done")


main()
