import numpy as np
import matplotlib.pyplot as plt
import random

def main():
    print("Hello world!")

def R_squared(func, data: np.ndarray):
    if data.shape[1] != 2:
        raise Exception("Invalid data input, not 2 columns")
    func_points = np.hstack((np.array(data[:, [0]]), func(data[:, [0]])))
    # Get sum of squares of residuals
    rss = np.sum(np.square(data[:,[1]] - func_points[:,[1]]))
    print(np.mean(data[:,[1]]))
    print(np.square(data[:,[1]] - np.mean(data[:,[1]])))
    # Get total sum of squares
    tss = np.sum(np.square(data[:,[1]] - np.mean(data[:,[1]])))
    return 1 - rss/tss

def MSE(func, data: np.ndarray):
    if data.shape[1] != 2:
        raise Exception("Invalid data input, not 2 columns")
    func_points = np.hstack((np.array(data[:, [0]]), func(data[:, [0]])))
    return func_points[:,[1]] - data[:,[1]]

# Output from a polynomial with input x and weights w
def polynomial(x: np.ndarray, w: np.ndarray):
    if len(x.shape) > 1 and x.shape[1] > 1:
        raise Exception(f"Invalid data input, not 1 column: {x.shape}")
    elif w.shape[0] != 1:
        raise Exception(f"Invalid weights input, not 1 column: {w.shape}")
    return np.power(x, np.array([[i for i in range(w.shape[1])]])).dot(w.T)

def plot_func(func, x):
    plt.plot(func(x))

# Get co-efficients of a fitted polynomial to some data
# data: n x 2 ndarray
# deg: int
def fit_poly(data: np.ndarray, deg = 2):
    if len(data.shape) > 1 and data.shape[1] != 2:
        raise Exception("Invalid data input, not 2 columns")
    poly = np.polynomial.Polynomial(np.arange(0, deg))
    return np.array([poly.fit(data[:, 0], data[:,1], deg).convert().coef])


def test():
    # generate random values
    x: [float] = [-23.5]
    y: [float] = [1]
    random.seed(13)
    for i in range(1,48):
        rnd = round(random.random()*10) - 5
        x.append(i-23.5)
        y.append(y[i - 1] + rnd)
        last = y[i - 1] + rnd
    example_points = np.array([x,y]).T
    print(example_points)

    # Get polynomial points

    w2 = fit_poly(example_points, 10)
    print(w2)
    f1 = plt.figure()
    ax = plt.gca()
    ax.set_ylim([-20,40])
    plt.plot(example_points)
    poly_points = example_points.copy()
    poly_points[:, [1]] = polynomial(example_points[:, [0]], w2)
    plt.plot(poly_points)
    #plot_func(polynomial, example_points[:,[0]])
    plt.show()

if __name__ == "__main__":
    test()

