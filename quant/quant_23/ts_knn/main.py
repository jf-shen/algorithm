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
from sample_builder import SampleBuilder
from six import StringIO
import glob
from factor import KnnMinuteFactor, KnnFactor


# 初始化函数
def initialize(context):
    set_slippage(FixedSlippage(0.02))  # 滑点高（不设置滑点的话用默认的0.00246）
    set_benchmark('000300.XSHG')  # 沪深300
    set_option('use_real_price', True)  # 用真实价格交易
    # set_option("avoid_future_data", True)
    log.set_level('order', 'error')  # 过滤order中低于error级别的日志
    warnings.filterwarnings("ignore")

    # 选股参数
    g.freq = 3
    g.pattern_len = 7
    g.days = 0
    g.stock_num = 5

    # 止损参数
    g.stop_trading = False
    g.max_return = 0

    logger.set_log_level(2)

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

    if context.portfolio.returns < -1:  # -0.2: # -0.2:
        print("stop loss is triggered!")
        for stock in context.portfolio.positions.keys():
            order_target_value(stock, 0)
        g.stop_trading = True

    if max_drawback > 1:  # 0.25:
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

        f1 = KnnFactor(context, g)
        f1.set_data_path('data/ts_knn_20230116/')
        f1.fill_last_reward()
        f1.run()
        f1.write_data()

        factors = ['open_f_', 'close_f_', 'high_f_', 'low_f_', 'volume_f_', 'avg_f_']
        factor_return_dict = f1.get_recent_factor_return(context, factors=factors)
        log.info('近期因子收益 = ', factor_return_dict)

        best_factor_name = None
        best_factor_return = -999

        for factor_name, factor_return in factor_return_dict.items():
            if factor_return is not None and factor_return > best_factor_return:
                best_factor_name = factor_name
                best_factor_return = factor_return

        if best_factor_name is not None:
            buy_stock_info = f1.candidate_df.sort_index(by=best_factor_name, ascending=False).head(5)
            buy_stocks = buy_stock_info['code'].values
            log.info('今日自选股:%s' % buy_stocks)

            log.info('筛选因子: %s' % best_factor_name)
            log.info('分数 = ', buy_stock_info[['code', best_factor_name]].values)

            adjust_position(context, buy_stocks)
    g.days += 1


def adjust_position(context, buy_stocks):
    for stock in context.portfolio.positions:
        if stock not in buy_stocks:
            order_target(stock, 0)

    position_count = len(context.portfolio.positions)
    if g.stock_num > position_count:
        value = context.portfolio.cash / (g.stock_num - position_count)
        for stock in buy_stocks:
            if stock not in context.portfolio.positions:
                print(stock, value)
                order_target_value(stock, value)
                if len(context.portfolio.positions) == g.stock_num:
                    break



