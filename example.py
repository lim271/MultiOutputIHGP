import numpy as np
from moihgp import MOIHGPOnlineLearning
from time import time
import matplotlib.pyplot as plt



dt = 0.1
gamma = 0.99
windowsize = 3

if __name__=='__main__':
    v1 = np.array([1.1, 0.9])
    v2 = np.array([0.9, 1.1])
    v3 = np.array([-1.1, -0.9])
    v4 = np.array([-0.9, -1.1])
    p11 = [np.array([-1.1, -0.9])]
    p12 = [np.array([-0.9, -1.1])]
    p21 = [np.array([1.1, 0.9])]
    p22 = [np.array([0.9, 1.1])]
    v11 = []
    v12 = []
    v21 = []
    v22 = []
    for t in range(10):
        v11.append(v1 + 0.1 * np.random.randn(2))
        v12.append(v2 + 0.1 * np.random.randn(2))
        v21.append(v3 + 0.1 * np.random.randn(2))
        v22.append(v4 + 0.1 * np.random.randn(2))
        p11.append(p11[-1] + v11[-1] * dt)
        p12.append(p12[-1] + v12[-1] * dt)
        p21.append(p21[-1] + v21[-1] * dt)
        p22.append(p22[-1] + v22[-1] * dt)
    v1 = np.array([0.9, 1.1])
    v2 = np.array([1.1, 0.9])
    v3 = np.array([-0.9, -1.1])
    v4 = np.array([-1.1, -0.9])
    for t in range(10):
        v11.append(v1 + 0.1 * np.random.randn(2))
        v12.append(v2 + 0.1 * np.random.randn(2))
        v21.append(v3 + 0.1 * np.random.randn(2))
        v22.append(v4 + 0.1 * np.random.randn(2))
        p11.append(p11[-1] + v11[-1] * dt)
        p12.append(p12[-1] + v12[-1] * dt)
        p21.append(p21[-1] + v21[-1] * dt)
        p22.append(p22[-1] + v22[-1] * dt)
    data = np.hstack([p11, p12, p21, p22])
    _, num_output = data.shape
    num_latent = num_output - 1
    gp = MOIHGPOnlineLearning(dt, num_output, num_latent, gamma, windowsize, False)
    yhat = []
    for y in data:
        tic = time()
        yhat.append(gp.step(y))
        toc = time()
        print("elapsed time per step:", toc - tic)
    yhat = np.array(yhat)
    plt.figure(1)
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(data[:, 2], data[:, 3])
    plt.scatter(data[:, 4], data[:, 5])
    plt.scatter(data[:, 6], data[:, 7])
    plt.plot(yhat[:, 0] , yhat[:, 1])
    plt.plot(yhat[:, 2] , yhat[:, 3])
    plt.plot(yhat[:, 4] , yhat[:, 5])
    plt.plot(yhat[:, 6] , yhat[:, 7])
    plt.show()
