import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray


plt.rcParams['figure.dpi'] = 100

def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


def formatPrice(data: ndarray) -> str:
    return f"{data}" + "PLN"


def getState(data: ndarray, t: int, n: int) -> ndarray:

    d = t - n + 1

    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    res = []

    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])


def plot_decisions(data, states_buy, states_sell, profit):
    fig = plt.figure(figsize = (15,5))

    plt.plot(data, color = 'r', lw = 2.)
    plt.plot(data, "^",
             markersize = 10,
             color = 'm',
             label = "Buying signal",
             markevery = states_buy)

    plt.plot(data, "v",
             markersize = 10,
             color = 'k',
             label = "Selling signal",
             markevery = states_sell)

    plt.title(f"Total Gains: {np.round(profit, 2)}")
    plt.legend()
    plt.show()




