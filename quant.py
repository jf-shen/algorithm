import numpy as np
import matplotlib.pyplot as plt
import seaborn 


## 获取涨幅
def get_diff(x, relative = True):
    diff = x[1:] - x[:-1]
    return diff / x[1:] if relative else diff 

## 获取序列自相关性矩阵
def get_auto_corr(x, dt = 1):
    corr = np.corrcoef(x[dt:], x[:-dt])
    return corr[0,1]

## 获取涨幅自相关性
def get_incr_corr(x):
    return get_auto_corr(get_diff(x))

def show_incr_dist(x):
    plt.hist(get_diff(x), bins = 20)


## 回测类
class Context:
    def __init__(self, price):
        self.price = np.array(price) 
        self.T = len(self.price)
        
        self.position = np.zeros_like(price)
        self.base_position = np.ones_like(price)
        
    def get_profit(self, t = None):    
        if t is None:
            t = self.T
            
        if t <= 0:
            return 0 
        
        diff = get_diff(self.price[:t])
        position = self.position[:t-1]
        
        return diff.dot(position)
    
    def get_profit_sequence(self, position):
        price = self.price
        profit = np.zeros_like(price)
        for t in range(self.T - 1):
            profit[t + 1] = profit[t] + position[t] * (price[t + 1] - price[t])
        return profit
    
    def plot_profit(self):
        profit_sequence = self.get_profit_sequence(self.position)
        base_profit_sequence = self.get_profit_sequence(self.base_position)
        
        plt.plot(profit_sequence, label = 'strategy')
        plt.plot(base_profit_sequence, label = 'base')
        plt.legend()
        plt.show()
    
if __name__ == '__main__':   
    context = Context(x)

    for t in range(strategy.T):
        if t >= 1: 
            if context.price[t] > context.price[t - 1]:
                context.position[t] = 2
            else:
                context.position[t] = 0

    context.plot_profit()
        
        