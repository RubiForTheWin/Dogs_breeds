from sklearn.preprocessing import StandardScaler
import numpy as np


def main():
    a = np.array([1, 2, 3]).reshape(-1, 1)
    sc = StandardScaler()
    a_normed = sc.fit_transform(a)
    print(a_normed)


main()