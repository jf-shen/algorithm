"""
Reference: http://krasserm.github.io/2018/03/21/bayesian-optimization/?spm=ata.13261165.0.0.4bc32783ZqT8sz
"""

import numpy as np
from math import erf


# RBF Kernel
def rbf(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-1/2 * x * x)


# Cumulative distribution function for the standard normal distribution
def norm_cdf(x):
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0


class BayesSearch:
    def __init__(self, kernel=rbf, mu_0=0, std_0=1):
        """
        @param kernel: kernel function of Gaussian Process
        @param std_0: standard deviation of prior Gaussian Process, i.e. win_size for kernel function
        @param mu_0: mean of Gaussian Process
        """
        assert std_0 > 0, "std <= 0"

        def k(x, y=None):
            if y is None:
                return kernel(x / std_0) / std_0
            else:
                x = np.array(x)
                y = np.array(y)
                return k(np.linalg.norm(x - y))

        self.k = k
        self.mu_0 = mu_0
        self.std_0 = std_0

        self.x = None
        self.y = None
        self.data_num = None
        self.Sigma_0 = None
        self.A = None

    def fit(self, x, y, noise_std=None):
        """
        @param x: data points, shape = [n, d]
        @param y: observations, shape = [n]
        @param noise_std: standard deviation of white noise for each observation (default: std_0 / 10)

        data_num: number of data points
        Sigma_0: covariance function for X[0], ..., X[n-1]
        """

        if noise_std is None:
            noise_std = self.std_0 / 10

        self.x = np.array(x)
        self.y = np.array(y)

        self.data_num = len(self.y)
        self.Sigma_0 = np.zeros([data_num, data_num])

        for i in range(data_num):
            for j in range(data_num):
                self.Sigma_0[i, j] = self.k(x[i], x[j])

        for i in range(data_num):
            self.Sigma_0[i, i] += noise_std ** 2

        self.A = np.linalg.inv(self.Sigma_0)

    def get_dist(self, x):
        """
        @param x: data point to be investigate
        @return [mean, std]
        """
        b = np.zeros(self.data_num)
        for i in range(self.data_num):
            b[i] = self.k(x, self.x[i])

        mean = self.mu_0 + self.A.dot(self.y).dot(b)
        var = self.std_0 - self.A.dot(b).dot(b)
        std = np.sqrt(var)

        return [mean, std]

    def eval(self, x, eps=1e-2):
        """
        @param x: data point to be investigate
        @return score: value of using x
        """

        mean, std = self.get_dist(x)
        max_y = max(self.y)

        delta = mean - max_y - eps
        expected_improvement = std * rbf(delta / std) + delta * norm_cdf(delta / std)

        return expected_improvement


if __name__ == '__main__':
    dim = 4
    data_num = 100
    x = np.random.normal(0, 1, [data_num, dim])
    y = np.random.normal(0, 1, data_num)

    bs = BayesSearch(kernel=rbf, mu_0=0, std_0=1)
    bs.fit(x, y)

    z = np.random.normal(0, 1, dim)
    print(bs.eval(z))
