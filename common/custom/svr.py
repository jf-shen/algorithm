"""
online dual SVR
reference:
Foundation of Machine Learning, 2th-edition, Chapter 11.4, P291-Figure 11.8
"""

import numpy as np


def rbf(x, y):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-np.sum((x - y) ** 2) / 2)


kernel_fn = {
    'rbf': rbf
}


def fit_online_dual_svr(X, Y, kernel='rbf', epsilon=0.1, eta=1e-2):
    K = kernel_fn[kernel]

    T, d = X.shape
    alpha = np.zeros(T)
    alpha_ = np.zeros(T)

    x_list = []

    for t in range(T):
        x = X[t]
        y = Y[t]

        y_hat = 0
        for x_s in x_list:
            y_hat += (alpha_ - alpha) * K(x_s, x)

        alpha_ = alpha_ + min(max(eta * (y - y_hat - epsilon), -alpha_), C - alpha_)
        alpha = alpha + min(max(eta * (y_hat - y - epsilon), -alpha), C - alpha)

    def svr_fn(x):
        res = 0
        for i in range(T):
            res += (alpha_ - alpha) * K(x_t, x)
        return res

    return svr_fn
