import numpy as np
from moihgp import MOIHGPOnlineLearning
from time import time
import matplotlib.pyplot as plt



dt = 0.1
gamma = 0.9
l1_reg = 0.0
windowsize = 2

if __name__=='__main__':
    v1 = np.array([1.1, 0.9])
    v2 = np.array([-0.9, -1.1])
    p11 = [np.array([-1.1, -0.9])]
    p12 = [np.array([-0.9, -1.1])]
    p21 = [np.array([1.1, 0.9])]
    p22 = [np.array([0.9, 1.1])]
    v11 = []
    v12 = []
    v21 = []
    v22 = []
    for t in range(20):
        v11.append(v1 + 0.3 * np.sin(t) + 0.1 * np.random.randn(2))
        v12.append(v1 + 0.3 * np.cos(t) + 0.1 * np.random.randn(2))
        v21.append(v2 + 0.3 * np.sin(0.3*t) + 0.1 * np.random.randn(2))
        v22.append(v2 + 0.3 * np.cos(0.3*t) + 0.1 * np.random.randn(2))
        p11.append(p11[-1] + v11[-1] * dt)
        p12.append(p12[-1] + v12[-1] * dt)
        p21.append(p21[-1] + v21[-1] * dt)
        p22.append(p22[-1] + v22[-1] * dt)
    data = np.hstack([p11, p12, p21, p22])
    _, num_output = data.shape
    num_latent = round(num_output / 2)
    gp = MOIHGPOnlineLearning(dt, num_output, num_latent, gamma=gamma, l1_reg=l1_reg, windowsize=windowsize, threading=False)
    yhat = []
    for y in data:
        tic = time()
        yhat.append(gp.step(y))
        toc = time()
        print("elapsed time per step:", toc - tic)
        Cov = gp.covariance
        Corr = np.eye(num_output//2)
        for i in range(num_output//2):
            ii = slice(i*2, (i+1)*2)
            for j in range(num_output//2):
                jj = slice(j*2, (j+1)*2)
                Corr[i, j] = np.sign(np.trace(Cov[ii, jj])) * np.sqrt(np.trace(np.linalg.inv(Cov[ii, ii]) @ Cov[ii, jj] @ np.linalg.inv(Cov[jj, jj]) @ Cov[jj, ii])/2)
        print(Corr)
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
