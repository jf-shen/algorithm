# 克隆自聚宽文章：https://www.joinquant.com/post/10778
# 标题：【量化课堂】机器学习多因子策略
# 作者：JoinQuant量化课堂

import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.svm import SVR  
from sklearn.model_selection import GridSearchCV  
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import jqdata

def initialize(context):
    set_params()
    set_backtest()
    run_daily(strategy, 'every_bar')
    
def set_params():
    g.days = 0
    g.stockNum = 10
    g.corr_count = 200 

    g.candidate_set_refresh_rate = 10
    g.corr_dict_refresh_rate = g.candidate_set_refresh_rate 

    g.candidate_set = []
    g.corr_dict = {} 
    
def set_backtest():
    set_benchmark('000001.XSHG')
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    
def strategy(context):
    candidateSet = get_candidate_set(context)
    score_dict = rank(context, candidateSet)

    trade(context, candidateSet, score_dict)
    g.days += 1

def get_candidate_set(context):
    if g.days % g.candidate_set_refresh_rate != 0:
        return g.candidate_set

    X, Y = get_training_data(context)

    model = sklearn.linear_model.ElasticNet(alpha=1, l1_ratio=0.5)
    model.fit(X,Y)

    factor = Y - pd.DataFrame(model.predict(X), index = Y.index, columns = ['log_mcap'])    ## log(真实市值) - log(预测市值)
    factor = factor.sort_index(by = 'log_mcap')

    candidateSet = list(factor.index[:g.stockNum])

    g.candidate_set = candidateSet
    return candidateSet

def rank(context, candidateSet):

    end_date = context.current_dt - datetime.timedelta(days=0.5) 
    price = get_price(
        candidateSet, 
        start_date=None,   # 使用count 
        end_date=end_date, 
        frequency='daily', 
        fields=['close'], 
        skip_paused=False, 
        fq='pre', 
        count=g.corr_count
    )

    corr_dict = {}
    for index, seq in zip(price['close'], price['close'].values.T):
        corr_dict[index] = get_incr_corr(seq)
    g.corr_dict = corr_dict

    l_price = price['close'].iloc[-1,:]
    ll_price = price['close'].iloc[-2,:]


    trend = (l_price - ll_price) / ll_price
    trend_dict = dict(zip(trend.index, trend))
    
    score_dict = {} 
    for index, incr in trend_dict.items():
        # print(index)
        corr = corr_dict[index]
        if np.isfinite(corr):
            score_dict[index] = corr * incr
    
    return score_dict

def trade(context, candidateSet, score_dict):
    dataSet = [] 
    for index in candidateSet:
        if score_dict.get(index) is not None and score_dict[index] > 0:
            dataSet.append(index)

    print(dataSet)

    position_set = set(context.portfolio.positions.keys())
    target_set = set(dataSet)

    buy_set = target_set - position_set
    sell_set = position_set - target_set
    
    
    for stock in sell_set:
        order_target_value(stock, 0)
    
    if len(buy_set) > 0:
        cash = context.portfolio.available_cash / len(buy_set)
    
    for stock in buy_set:
        order_target_value(stock, cash)


def get_training_data(context):
    sample = get_index_stocks('000001.XSHG', date = None)
    q = query(
              valuation.code,
              valuation.market_cap, 
              balance.total_assets - balance.total_liability,  # 净资产
              balance.total_assets / balance.total_liability,  # 资产负债比
              income.net_profit, 
              income.net_profit + 1, 
              indicator.inc_revenue_year_on_year, 
              balance.development_expenditure
        ).filter(
            valuation.code.in_(sample)
        )
        
    df = get_fundamentals(q, date = None)

    df.columns = ['code', 'log_mcap', 'log_NC', 'LEV', 'NI_p', 'NI_n', 'g', 'log_RD']

    df['log_mcap'] = np.log(df['log_mcap'])
    df['log_NC'] = np.log(df['log_NC'])
    df['NI_p'] = np.log(np.abs(df['NI_p']))
    df['NI_n'] = np.log(np.abs(df['NI_n'][df['NI_n'] < 0]))
    df['log_RD'] = np.log(df['log_RD'])
    df.index = df.code.values

    del df['code']
    df = df.fillna(0)
    df[df > 10000] = 10000
    df[df < -10000] = -10000

    ### 行业编码
    industry_set = ['801010', '801020', '801030', '801040', '801050', '801080', '801110', '801120', '801130', 
              '801140', '801150', '801160', '801170', '801180', '801200', '801210', '801230', '801710',
              '801720', '801730', '801740', '801750', '801760', '801770', '801780', '801790', '801880','801890']

    factor_set = ['log_NC', 'LEV', 'NI_p', 'NI_n', 'g', 'log_RD'] 
    feature_list = factor_set.copy() 

    ### get dummy variable
    for i in range(len(industry_set)):
        industry = get_industry_stocks(industry_set[i], date = None)
        s = pd.Series([0]*len(df), index=df.index)
        s[set(industry) & set(df.index)]=1
        df[industry_set[i]] = s
        for factor in factor_set:
            fea_name = industry_set[i] + '_' + factor
            df[fea_name] = df[industry_set[i]] * df[factor]   # factor x industry
            feature_list.append(fea_name) 
        feature_list.append(industry_set[i])

        
    X = df[feature_list]
            
    Y = df[['log_mcap']]   ## 真实市值
    X = X.fillna(0)
    Y = Y.fillna(0)

    return X, Y

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



        
            