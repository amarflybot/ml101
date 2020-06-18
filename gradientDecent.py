import sys

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_function(theta, x, y):
    m = len(y)
    predictions = x.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost


def cost_function_logistic(theta, x, y):
    m = len(y)
    h = sigmoid(x @ theta)
    epsilon = 1e-5
    cost = (1 / m) * (((-y).T @ np.log(h + epsilon)) - ((1 - y).T @ np.log(1 - h + epsilon)))
    return cost


def gradient_decent(x, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        prediction = np.dot(x, theta)
        theta = theta - (1 / m) * learning_rate * (x.T.dot((prediction - y)))
        theta_history[it] = theta.T
        cost_history[it] = cost_function(theta, x, y)
    return theta, cost_history, theta_history


def gradient_decent_logistic(x, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        prediction = sigmoid(x @ theta)
        theta = theta - (learning_rate / m) * (x.T @ (prediction - y))
        theta_history[it] = theta.T
        cost_history[it] = cost_function_logistic(theta, x, y)
    return theta, cost_history, theta_history


def main():
    lr = 0.01
    n_iter = 1000
    x = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1)
    # Create the plot
    # plt.plot(x, sigmoid(x), "ob")
    # plt.show()
    theta = np.random.randn(2, 1)
    x_b = np.c_[np.ones((len(x), 1)), x]
    theta, cost_history, theta_history = gradient_decent_logistic(x_b, y, theta, lr, n_iter)

    print('Theta0:  {:0.3f},\nTheta1:   {:0.3f}'.format(theta[0][0], theta[1][0]))
    print('Final cost/MSE:      {:0.3f}'.format(cost_history[-1]))

    plt.plot(theta_history, cost_history, "ob")
    plt.show()


if __name__ == '__main__':
    main()
    sys.exit()
