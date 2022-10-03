# 克隆自聚宽文章：https://www.joinquant.com/post/10778
# 标题：【量化课堂】机器学习多因子策略
# 作者：JoinQuant量化课堂

import pandas as pd
import numpy as np
import math
from sklearn.svm import SVR  
from sklearn.model_selection import GridSearchCV  
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import jqdata

def initialize(context):
    set_params()
    set_backtest()
    run_daily(trade, 'every_bar')
    
def set_params():
    g.days = 0
    g.refresh_rate = 10
    g.stocknum = 10
    
def set_backtest():
    set_benchmark('000001.XSHG')
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    
def trade(context):
    if g.days % 10 == 0:
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
        
        ### get dummy variable
        for i in range(len(industry_set)):
            industry = get_industry_stocks(industry_set[i], date = None)
            s = pd.Series([0]*len(df), index=df.index)
            s[set(industry) & set(df.index)]=1
            df[industry_set[i]] = s
            df[industry_set[i] + '_log_RD'] = df[industry_set[i]] * df['log_RD']  
            
        X = df[['log_NC', 'LEV', 'NI_p', 'NI_n', 'g', 'log_RD','801010', '801020', '801030', '801040', '801050', 
                '801080', '801110', '801120', '801130', '801140', '801150', '801160', '801170', '801180', '801200', 
                '801210', '801230', '801710', '801720', '801730', '801740', '801750', '801760', '801770', '801780', 
                '801790', '801880', '801890',
                '801010_log_RD','801020_log_RD','801030_log_RD','801040_log_RD','801050_log_RD','801080_log_RD',
                '801110_log_RD','801120_log_RD','801130_log_RD','801140_log_RD','801150_log_RD','801160_log_RD',
                '801170_log_RD','801180_log_RD','801200_log_RD','801210_log_RD','801230_log_RD','801710_log_RD',
                '801720_log_RD','801730_log_RD','801740_log_RD','801750_log_RD','801760_log_RD','801770_log_RD',
                '801780_log_RD','801790_log_RD','801880_log_RD','801890_log_RD']]
                
        Y = df[['log_mcap']]   ## 真实市值
        X = X.fillna(0)
        Y = Y.fillna(0)
        
        svr = SVR(kernel='rbf', gamma = 0.1) 
        model = svr.fit(X, Y)
        factor = Y - pd.DataFrame(svr.predict(X), index = Y.index, columns = ['log_mcap'])    ## log(真实市值) - log(预测市值)
        factor = factor.sort_index(by = 'log_mcap')
        
        position_set = set(context.portfolio.positions.keys())
        top_set = set(factor.index[:g.stocknum])
        
        sell = position_set - top_set
        buy = top_set - position_set
        
        for stock in sell:
            order_target_value(stock, 0)
        
        if len(buy) > 0:
            cash = context.portfolio.available_cash / len(buy)
        
        for stock in buy:
            order_target_value(stock, cash)
        
        g.days += 1
    else:
        g.days = g.days + 1    
        
        
            