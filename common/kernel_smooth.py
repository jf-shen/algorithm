import numpy as np


# ================ Kernel Functions ============== #
def rbf(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * x * x)


# ================ Linear Regression ============== #
def get_1d_lr_coeff(x, y, w=None):
    if w is None:
        w = np.ones_like(x)
    s_xx = np.sum(w * x * x)
    s_xy = np.sum(w * x * y)
    s_x = np.sum(w * x)
    s_y = np.sum(w * y)
    s = np.sum(w)

    k = (s * s_xy - s_x * s_y) / (s * s_xx - s_x * s_x)
    b = (s_y - k * s_x) / s
    return k, b


# ================ Kernel Smooth ============== #
class KernelSmooth:
    def __init__(self, win_size=1, kernel=rbf):
        self.k = lambda x: kernel(x / win_size) / win_size

    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def predict(self, t):
        t = np.array(t)

        x = np.tile(self.x, [len(t),1])
        y = np.tile(self.y, [len(t),1])
        t = np.tile(t, [len(x),1]).T

        w = self.k(x-t)
        total_w = np.tile(np.sum(w, axis=1), [len(x), 1]).T
        w = w / total_w
        return np.sum(y*w, axis=1)


# ================ Local Linear Regression ============== #
class LLR:
    def __init__(self, win_size=1, kernel=rbf, x_scale = None):
        self.k = lambda x: kernel(x / win_size) / win_size
        self.x_scale = x_scale

        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def predict(self, t):
        if np.isscalar(t):
            if self.x_scale == 'log':
                w = self.k(np.log(self.x) - np.log(t))
                k, b = get_1d_lr_coeff(np.log(self.x), self.y, w)
                return k * np.log(t) + b
            else:
                w = self.k(self.x - t)
                k, b = get_1d_lr_coeff(self.x, self.y, w)
                return k * t + b
        else:
            return np.array([self.predict(t_i) for t_i in t])


# ================== Kernel Density Estimation ================== #
class KernelDensityEstimation:
    def __init__(self, win_size=1, kernel=rbf):
        self.k = lambda x: kernel(x / win_size) / win_size
        self.x = None

    def fit(self, x):
        self.x = np.array(x)

    def predict(self, t):
        t = np.array(t)
        x = np.tile(self.x, [len(t),1])
        t = np.tile(t, [len(self.x), 1]).T
        return np.sum(self.k(x - t), axis=1) / len(x)
