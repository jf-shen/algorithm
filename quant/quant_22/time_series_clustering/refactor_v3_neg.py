# 克隆自聚宽文章：https://www.joinquant.com/post/38623
# 标题：自适应量化终极算法2.0 （全新升级）
# 作者：JinnyKoo

# 导入函数库
from jqlib.technical_analysis import *
from jqdata import *
import warnings
import numpy as np
import pandas as pd
import time
import math
import datetime
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
    g.days = 0
    g.stock_num = 5

    # 止损参数
    g.stop_trading = False
    g.max_return = 0

    # 手续费参数
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0005, close_commission=0.0005, min_commission=5),
                   type='stock')
    # 设置交易时间
    run_daily(main, time='9:35', reference_security='000300.XSHG')


def update_status(context):
    if g.stop_trading:
        return

    g.max_return = max(context.portfolio.returns, g.max_return)
    max_drawback = 1 - (1 + context.portfolio.returns) / (1 + g.max_return)

    if context.portfolio.returns < -0.5:
        print("stop loss is triggered!")
        for stock in context.portfolio.positions.keys():
            order_target_value(stock, 0)
        g.stop_trading = True

    if max_drawback > 0.6:
        print("stop win is triggered!")
        for stock in context.portfolio.positions.keys():
            order_target_value(stock, 0)
        g.stop_trading = True

    prefix = '%s (%i%%%i=%i)' % (context.current_dt, g.days, g.freq, g.days % g.freq)
    log.info('[%s] return = %.4f, max_drawback = %.4f' % (
        prefix, context.portfolio.returns, max_drawback))


# 开盘时运行函数
def main(context):
    update_status(context)

    if g.days % g.freq == 0 and (not g.stop_trading):
        initial_list = get_stock_list(context)

        all_patterns = collect_patterns(
            stocks=initial_list,
            predict_window=g.freq,
            pattern_length=7,
            end_date=context.previous_date,
            neg=False
        )

        all_patterns_neg = collect_patterns(
            stocks=initial_list,
            predict_window=g.freq,
            pattern_length=7,
            end_date=context.previous_date,
            neg=True
        )

        print("collect pattern finish, pattern num = %i" % len(all_patterns))

        stocks_to_buy = pattern_match(stocks=initial_list,
                                      patterns=all_patterns,
                                      pattern_length=7,
                                      end_date=context.previous_date,
                                      dis=1)

        stocks_to_sell = pattern_match(stocks=initial_list,
                                       patterns=all_patterns_neg,
                                       pattern_length=7,
                                       end_date=context.previous_date,
                                       dis=1)

        final_score = {k: v - stocks_to_sell[k] / len(all_patterns_neg) * len(all_patterns) for k, v in
                       stocks_to_buy.items()}
        check_out_list = [k for k, v in sorted(final_score.items(), key=lambda kv: kv[1])][::-1][:5]

        for k in check_out_list:
            print(k, final_score.get(k), stocks_to_buy.get(k), stocks_to_sell.get(k))

        check_out_list = list(filter(lambda k: final_score[k] > 0, check_out_list))

        log.info('今日自选股:%s' % check_out_list)
        adjust_position(context, check_out_list)
    g.days += 1


def filter_name(name):
    return ('ST' not in name) and ('*' not in name) and ('退' not in name)


def filter_code(code):
    return (not code.startswith('300')) and (not code.startswith('688')) and (not code.startswith('002'))


def get_stock_list(context):
    # 获取选股列表并过滤掉:st,st*,退市,涨停,跌停,停牌
    stock_df = get_all_securities()
    stock_df['code'] = stock_df.index
    stock_df = stock_df[stock_df['display_name'].apply(filter_name)]
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


def bi_func(p, a, bi_op):
    res = Pattern()
    if isinstance(a, Pattern):
        res.open = bi_op(p.open, a.open)
        res.close = bi_op(p.close, a.close)
        res.high = bi_op(p.high, a.high)
        res.low = bi_op(p.low, a.low)
        res.volume = bi_op(p.volume, a.volume)
    else:
        res.open = bi_op(p.open, a)
        res.close = bi_op(p.close, a)
        res.high = bi_op(p.high, a)
        res.low = bi_op(p.low, a)
        res.volume = bi_op(p.volume, a)
    return res


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

    def __add__(self, p):
        return bi_func(self, p, lambda x, y: x + y)

    def __div__(self, p):
        return bi_func(self, p, lambda x, y: x / y)

    def __mul__(self, p):
        return bi_func(self, p, lambda x, y: x * y)

    def __sub__(self, p):
        return bi_func(self, p, lambda x, y: x - y)

    def apply_map(self, fn):
        res = Pattern()
        res.open = fn(self.open)
        res.close = fn(self.close)
        res.high = fn(self.high)
        res.low = fn(self.low)
        res.volume = fn(self.volume)
        return res

    def __repr__(self):
        print('Patterns:')
        print('open = ', self.open)
        print('close = ', self.close)
        print('high = ', self.high)
        print('low = ', self.low)
        print('volume = ', self.volume)
        return ''


def get_lr_coef(prices):
    x = np.arange(len(prices))
    y = prices
    return (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)


def trend(prices, index):
    # 1. abnormal test
    if np.isnan(prices).any() or prices[0] == 0:
        return -1

    price_incr = (prices[-1] - prices[0]) / prices[0]
    index_incr = (index[-1] - index[0]) / index[0]

    # 2. final return test
    if price_incr < 0.15:
        return -1
    # 3. trend rank test
    if rankp(prices) < 0.99:
        return -1
    # 4. trend linear regression test
    if get_lr_coef(prices) <= 0:
        return -1
    # 5. price w.r.t index incr test
    if price_incr - index_incr < 0.1:
        return -1
    return 1


def trend_neg(prices, index):
    # 1. abnormal test
    if np.isnan(prices).any() or prices[0] == 0:
        return -1
    # 2. final return test
    if (prices[-1] - prices[0]) / prices[0] > -0.1:
        return -1
    # # 3. trend rank test
    # if irankp(prices) < 0.99:
    #     return -1
    # # 4. trend linear regression test
    # if get_lr_coef(prices) >= 0:
    #     return -1

    # # 5. index
    # price_incr = (prices[-1] - prices[0]) / prices[0]
    # index_incr = (index[-1] - index[0]) / index[0]

    # if price_incr - index_incr > -0.1:
    #     return -1

    return 1


def pattern_scan(df, predict_window, pattern_length, df_mean, neg=False):
    patterns = []
    prices = df['open'].tolist()
    predict_end = len(prices)
    while predict_end > predict_window + pattern_length:
        predict_start = predict_end - predict_window
        flag_p = trend(prices=prices[predict_start:predict_end], index=df_mean[predict_start:predict_end]) > 0
        flag_n = trend_neg(prices=prices[predict_start:predict_end], index=df_mean[predict_start:predict_end]) > 0
        flag = flag_n if neg else flag_p

        if flag:
            pattern = Pattern(pattern_start=predict_end - predict_window - pattern_length,
                              pattern_end=predict_end - predict_window,
                              predict_start=predict_start,
                              predict_end=predict_end)
            patterns.append(pattern)
            predict_end = pattern.pattern_start - 1
        else:
            predict_end -= 1
    return patterns


@timeit(name='collect_patterns')
def collect_patterns(stocks, predict_window, pattern_length, end_date, neg=False):
    all_patterns = []
    start_date = end_date - datetime.timedelta(days=365 * 3)

    df_mean = get_price(stocks,
                        start_date=start_date,
                        end_date=end_date,
                        frequency='daily',
                        fields=['open'],
                        skip_paused=False,
                        fq='pre').iloc[0, :, :].T.mean(axis=0)

    for stock in stocks:
        df = get_price(stock,
                       start_date=start_date,
                       end_date=end_date,
                       frequency='daily',
                       fields=None,
                       skip_paused=False,
                       fq='pre')

        df = df.iloc[::-1]
        patterns = pattern_scan(df, predict_window, pattern_length, df_mean, neg=neg)

        for pattern in patterns:
            pattern.get_seq(df)

        patterns = list(filter(lambda p: p.is_valid(), patterns))

        all_patterns += patterns

    print("pattern num = %i" % len(all_patterns))
    return all_patterns


def l2_dist(p1, p2):
    dist = lambda x1, x2: np.linalg.norm(x1 - x2)

    open_dist = dist(p1.open, p2.open)
    close_dist = dist(p1.close, p2.close)
    high_dist = dist(p1.high, p2.high)
    low_dist = dist(p1.low, p2.low)
    volume_dist = dist(p1.volume, p2.volume)
    distance = np.mean([open_dist, close_dist, high_dist, low_dist, volume_dist])

    return distance


def mean(patterns):
    return sum(patterns) / len(patterns)


def std(patterns):
    p_mean = mean(patterns)
    p_var = sum(map(lambda p: (p - p_mean) * (p - p_mean), patterns)) / len(patterns)
    p_std = p_var.apply_map(np.sqrt)
    return p_std


def normalize(patterns):
    p_mean = mean(patterns)
    p_std = std(patterns)

    for p in patterns:
        p = (p - p_mean) / (p_std + 1e-3)


@timeit(name='pattern_match')
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

        def rbf(x, sigma=0.1):
            x = np.array(x)
            return np.exp(- x ** 2 / (2 * sigma ** 2))

        stocks_to_buy[stock] = np.sum(rbf(dist_list))

    return stocks_to_buy


def adjust_position(context, buy_stocks):
    # order_value(g.bond,context.portfolio.available_cash)
    for stock in context.portfolio.positions:
        if stock not in buy_stocks:
            order_target(stock, 0)

    position_count = len(context.portfolio.positions)
    if g.stock_num > position_count:
        value = context.portfolio.cash / (g.stock_num - position_count)
        for stock in buy_stocks:
            if stock not in context.portfolio.positions:
                order_target_value(stock, value)
                if len(context.portfolio.positions) == g.stock_num:
                    break



