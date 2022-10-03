# 克隆自聚宽文章：https://www.joinquant.com/post/38623
# 标题：自适应量化终极算法2.0 （全新升级）
# 作者：JinnyKoo

# 导入函数库
import jqdata
from jqlib.technical_analysis import *
from jqdata import *
import warnings
from datetime import date
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import time
import math
import datetime
from datetime import date
from util import *


# 初始化函数
def initialize(context):
    # 滑点高（不设置滑点的话用默认的0.00246）
    set_slippage(FixedSlippage(0.02))
    # 沪深300
    set_benchmark('000300.XSHG')
    # 用真实价格交易
    set_option('use_real_price', True)
    # set_option("avoid_future_data", True)
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    warnings.filterwarnings("ignore")

    # 选股参数
    g.freq = 7
    g.count = 0
    g.position = 1  # 仓位
    g.stock_num = 5
    # 手续费
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0005, close_commission=0.0005, min_commission=5),
                   type='stock')

    # 设置交易时间
    run_daily(my_trade, time='9:45', reference_security='000300.XSHG')


# 开盘时运行函数
def my_trade(context):
    # 获取选股列表并过滤掉:st,st*,退市,涨停,跌停,停牌
    if g.count % g.freq == 0:
        initial_list = get_stock_list(context)

        all_patterns = collect_patterns(
            stocks=initial_list,
            threshold=0.1,
            predict_window=7,
            pattern_length=7,
            end_date=context.previous_date,
            K=5)

        print("collect pattern finish")

        # all_patterns = list(filter(lambda p: min(p.volume) > 0, all_patterns))
        centers, assigns = kmeans(all_patterns, k=5, iter_num=10, dist_fn=l2_dist, avg_fn=avg)

        stocks_to_buy = pattern_match(stocks=initial_list,
                                      patterns=centers,
                                      pattern_length=7,
                                      end_date=context.previous_date,
                                      dis=1)

        check_out_list = [k for k, v in sorted(stocks_to_buy.items(), \
                                               key=lambda item: item[1])][:5]

        for k in check_out_list:
            print(k, stocks_to_buy[k])

        log.info('今日自选股:%s' % check_out_list)
        adjust_position(context, check_out_list)
    g.count += 1


def filter_name(name):
    return ('ST' not in name) and ('*' not in name) and ('退' not in name)


def filter_code(code):
    return (not code.startswith('300')) and (not code.startswith('688')) and (not code.startswith('002'))


def get_stock_list(context):
    stock_df = get_all_securities()
    stock_df = stock_df[stock_df['display_name'].apply(filter_name)]
    stock_df['code'] = stock_df.index
    stock_df = stock_df[stock_df['code'].apply(filter_code)]

    initial_list = stock_df.index.tolist()

    panel = get_price(security=initial_list,
                      start_date=context.current_dt,
                      end_date=context.current_dt,
                      frequency='daily',
                      fields=['open', 'close', 'high', 'low', 'volume', 'money', 'avg',
                              'high_limit', 'low_limit', 'paused'],
                      skip_paused=False,
                      fq='pre')

    price_df = panel[:, 0, :]

    initial_list = [
        stock for stock in initial_list if not (
                (price_df.loc[stock].open == price_df.loc[stock].high_limit) or
                (price_df.loc[stock].open == price_df.loc[stock].low_limit) or
                (price_df.loc[stock].paused != 0)
        )]

    q = query(
        valuation.code,
        valuation.market_cap
    ).filter(
        valuation.code.in_(initial_list)
    ).order_by(
        valuation.market_cap.asc()
    )

    df = get_fundamentals(q)
    initial_list = list(df['code'])[:int(0.2 * len(list(df.code)))]

    return initial_list


class Pattern:
    def __init__(self, pattern_start=None, pattern_end=None, predict_start=None, predict_end=None):
        self.pattern_start = pattern_start
        self.pattern_end = pattern_end
        self.predict_start = predict_start
        self.predict_end = predict_end

        # pattern sequences
        self.open = None
        self.close = None
        self.high = None
        self.low = None
        self.volume = None

    def get_seq(self, df):
        if (self.pattern_start is not None) and (self.pattern_end is not None):
            df = df.iloc[self.pattern_start: self.pattern_end, :]
        self.open = (df.open / df.open[0]).astype('float64').values
        self.close = (df.close / df.close[0]).astype('float64').values
        self.high = (df.high / df.high[0]).astype('float64').values
        self.low = (df.low / df.low[0]).astype('float64').values
        self.volume = (df.volume / (df.volume[0] + 1)).astype('float64').values

    def is_valid(self):
        for seq in [self.open, self.close, self.high, self.low, self.volume]:
            if np.isnan(seq).any():
                return False
            if seq[0] < 0.01:
                return False
        return True

    def __add__(self, pt):
        res = Pattern()
        res.open = self.open + pt.open
        res.close = self.close + pt.close
        res.high = self.high + pt.high
        res.low = self.low + pt.low
        res.volume = self.volume + pt.volume
        return res

    def __div__(self, a):
        res = Pattern()
        res.open = self.open / a
        res.close = self.close / a
        res.high = self.high / a
        res.low = self.low / a
        res.volume = self.volume / a
        return res

    def __repr__(self):
        print('open = ', self.open)
        print('close = ', self.close)
        print('high = ', self.high)
        print('low = ', self.low)
        print('volume = ', self.volume)
        return ''


def trend(prices, threshold):
    if np.isnan(prices).any():
        return -1
    if (prices[-1] - prices[0]) / prices[0] < threshold:
        return -1;
    if rankp(prices) < 0.99:
        return -1

    x = np.array(range(len(prices))).reshape(-1, 1)
    y = prices
    reg = LinearRegression().fit(x, y)

    if y[0] == 0:
        return -1
    if reg.coef_[0] > 0:
        return 1
    else:
        return -1


def pattern_scan(df, threshold, predict_window, pattern_length):
    patterns = []
    prices = df['open'].tolist()
    volumes = df['volume'].tolist()
    predict_end = len(prices)
    while predict_end > predict_window + pattern_length:
        predict_start = predict_end - predict_window
        if (trend(prices[predict_start:predict_end], threshold) > 0):
            pattern = Pattern(pattern_start=predict_end - predict_window - pattern_length,
                              pattern_end=predict_end - predict_window,
                              predict_start=predict_start,
                              predict_end=predict_end)
            patterns.append(pattern)
            predict_end = pattern.pattern_start - 1
        else:
            predict_end -= 1
    return patterns


# 对保存的形态进行聚类操作， 使用聚类算法如DTW
def collect_patterns(stocks, threshold, predict_window, pattern_length, end_date, K=5):
    # ==================== SCAN ==================== #
    all_patterns = []
    start_date = end_date - datetime.timedelta(days=365 * 3)

    for stock in stocks:
        df = get_price(stock,
                       start_date=start_date,
                       end_date=end_date,
                       frequency='daily',
                       fields=None,
                       skip_paused=False,
                       fq='pre')

        df = df.iloc[::-1]
        patterns = pattern_scan(df, threshold, predict_window, pattern_length)

        for pattern in patterns:
            pattern.get_seq(df)

        patterns = list(filter(lambda p: p.is_valid(), patterns))

        all_patterns += patterns

    print("pattern num = %i" % len(all_patterns))
    return all_patterns


import numpy as np

a = None
b = None


def kmeans(X, k, iter_num=10, dist_fn=None, avg_fn=None):
    """
    Args:
        X: data = [x1, ..., xn]
        k: center num
        dist_fn: (x, y) -> dist
        avg_fn: [x1,..,xn] -> x_avg
    Return:
        centers: centers
        assigns: assignment of samples: {sample_index: center_index}
    """

    sample_num = len(X)
    assigns = np.random.choice(k, [sample_num])

    for i in range(iter_num):
        print('iter_num = %i' % i)
        centers = do_avg(X, assigns, avg_fn=avg_fn)
        assigns = do_assign(X, centers, dist_fn=dist_fn)
        print('center_num = %i, center_max = %.2f' % (len(centers), max(centers[0].volume)))

    return centers, assigns


def do_avg(X, assigns, avg_fn):
    centers = []
    cluster_indices = np.unique(assigns)
    for cluster_idx in cluster_indices:
        members = []
        for i, assign_idx in enumerate(assigns):
            if assign_idx == cluster_idx:
                members.append(X[i])
        centers.append(avg_fn(members))
    return centers


def do_assign(X, centers, dist_fn):
    k = len(centers)
    assigns = [None] * len(X)
    for sample_idx, x in enumerate(X):
        assign, min_dist = None, 1e10
        for center_idx in range(k):
            dist = dist_fn(x, centers[center_idx])
            if dist < min_dist:
                assign = center_idx
                min_dist = dist
        assigns[sample_idx] = assign
    return assigns


def avg(members):
    cnt = 0
    center = None
    for x in members:
        center = x if center is None else center + x
        cnt += 1
    return center / cnt


def l2_dist(p1, p2):
    dist = lambda x1, x2: np.linalg.norm(x1 - x2)

    open_dist = dist(p1.open, p2.open)
    close_dist = dist(p1.close, p2.close)
    high_dist = dist(p1.high, p2.high)
    low_dist = dist(p1.low, p2.low)
    volume_dist = dist(p1.volume, p2.volume)
    distance = (open_dist + close_dist + high_dist + low_dist) * volume_dist
    # distance = np.sqrt(open_dist ** 2 + close_dist **2 + high_dist **2 + low_dist **2 + volume_dist **2)

    return distance


def dtw_dist(p1, p2):
    dtw = lambda s1, s2: DTWDistance(s1, s2, w=5)

    open_dist = dtw(p1.open, p2.open)
    close_dist = dtw(p1.close, p2.close)
    high_dist = dtw(p1.high, p2.high)
    low_dist = dtw(p1.low, p2.low)
    volume_dist = dtw(p1.volume, p2.volume)

    return open_dist + close_dist + high_dist + low_dist + volume_dist


def pattern_match(stocks, patterns, pattern_length, end_date, dis):
    stocks_to_buy = {}
    for stock in stocks:
        df = get_price(stock,
                       count=pattern_length,
                       end_date=end_date,
                       frequency='daily', fields=None, skip_paused=False, fq='pre')
        df = df.iloc[::-1]

        current_shape = df.iloc[:pattern_length]
        df_features = current_shape

        pattern = Pattern()
        pattern.get_seq(df_features)
        if not pattern.is_valid():
            continue

        dist_list = []
        for p in patterns:
            distance = l2_dist(p, pattern)
            dist_list.append(distance)

        if np.min(dist_list) < dis:
            stocks_to_buy[stock] = np.min(dist_list)

    return stocks_to_buy


def adjust_position(context, buy_stocks):
    # order_value(g.bond,context.portfolio.available_cash)
    for stock in context.portfolio.positions:
        if stock not in buy_stocks:
            order_target(stock, 0)

    position_count = len(context.portfolio.positions)
    if g.stock_num > position_count:
        value = context.portfolio.cash * g.position / (g.stock_num - position_count)
        for stock in buy_stocks:
            if stock not in context.portfolio.positions:
                order_target_value(stock, value)
                if len(context.portfolio.positions) == g.stock_num:
                    break



