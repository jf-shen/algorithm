# graph

import pandas as pd
import numpy as np
import math
import sklearn
import jqdata
import json
from sklearn.svm import SVR
import os
import gc

from sample_builder import SampleBuilder
from feature_generator import FeatureGenerator
from signal_generator import SignalGenerator
from pipeline import feature_dt_list, label_dt_list
from util import Stream


def initialize(context):
    set_params()
    set_backtest()
    run_daily(main, 'every_bar')


def set_params():
    g.days = 0
    g.cycle = 5
    g.index_code = '000001.XSHG'
    g.past_df = None
    g.sig_num = 10


def set_backtest():
    set_benchmark('000001.XSHG')
    set_option('use_real_price', True)
    log.set_level('order', 'error')


def main(context):
    if g.days % g.cycle == 0:
        score_df = get_signal(context)
        trade(context, score_df)
    g.days += 1


def get_signal(context):
    sb = SampleBuilder(index_code=g.index_code)
    fg = FeatureGenerator()
    fg.set_feature_dt_list(feature_dt_list)
    fg.set_label_dt_list(label_dt_list)

    sb.set_pipeline()
    sb.set_fg(fg)

    df_train = sb.build_sample(
        count=720,
        end_date=context.current_dt - datetime.timedelta(5),
        sample_dt=10,
        mode='train')

    df_predict = sb.build_sample(
        start_date=context.current_dt,
        end_date=context.current_dt,
        sample_dt=1,
        mode='predict')

    sg = SignalGenerator(df_train)
    sg.training_set = sg.df
    sg.set_label('_price_incr_5d')

    params = {'max_depth': 5, 'eta': 1, 'objective': 'reg:linear', 'silent': 0}

    for i in range(g.sig_num):
        print('producing signal %i' % i)
        sg.set_features(sampling_rate=0.5)
        sg.fit_xgboost(num_round=100, params=params)
        df_predict['pred_%i' % i] = sg.predict_xgboost(df_predict)
        gc.collect()

    return df_predict


def trade(context, df):
    lst = Stream(range(g.sig_num)).map(lambda i: 'pred_%i' % i).tolist()
    df['rkscore'] = df[lst].applymap(lambda x: 1 if x > 0 else -1).mean()
    df = df.sort_index(by='rkscore', ascending=False)
    df = df.iloc[:5, :]

    position_set = set(context.portfolio.positions.keys())
    target_set = set(df['code'].values.tolist())

    buy_set = target_set - position_set
    sell_set = position_set - target_set

    for stock in sell_set:
        order_target_value(stock, 0)

    if len(buy_set) > 0:
        cash = context.portfolio.available_cash / len(buy_set)

    for stock in buy_set:
        order_target_value(stock, cash)
    return None 

