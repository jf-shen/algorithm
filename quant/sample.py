import numpy as np 
import pandas as pd 
import datetime
from quant import * 
from dataset import DataSet
import sklearn
from sklearn import linear_model 


def build_sample(start_date,
                 end_date,
                 count,
                 frequency,
                 sample_dt,
                 ts_len, 
                 label_dt_list, 
                 feature_dt_list,
                 index_code,
                 beta_count=300,
                 verbose=True):
    
    ## ================  get price ================== ## 
    stocks = get_index_stocks(index_code, date=end_date)

    df= get_price(stocks,  
              start_date=start_date, 
              end_date=end_date, 
              frequency=frequency, 
              fields=['close'], 
              skip_paused=False, 
              fq='pre', 
              count=count)

    df = df['close'].T

    df = get_display_name(df, set_index=True)       # set stock_name as index
    df.index.name = 'stock_name'

    df = diff_df(df, dt=1, remove_init_price=True)  #  stock price -> stock price increase 
    df = df.fillna(0)
    df = format_df_date(df)  # set column date format as 'yyyymmdd' 

    training_set = disassemble_time_series(df=df,
                               label_dt_list=label_dt_list,
                               feature_dt_list=feature_dt_list, 
                               start_idx=ts_len,
                               sample_dt=sample_dt,
                               label_name_format='_price_incr_%id',
                               feature_name_format = 'price_incr_d%i_',
                               prefix='price',
                               verbose=verbose)


    ## ================= Join index increase ===============  ##     

    index_price = get_price([index_code],  
              start_date=start_date, 
              end_date=end_date, 
              frequency=frequency, 
              fields=['close'], 
              skip_paused=False, 
              fq='pre', 
              count=count)

    index_price = index_price['close'].T

    index_price = diff_df(index_price, dt=1, remove_init_price=True)  # index price -> index price increase 
    index_price = format_df_date(index_price)   # set column date format as 'yyyymmdd' 


    index_incr_df = disassemble_time_series(df=index_price,
                                            start_idx=ts_len, 
                                            sample_dt=sample_dt,
                                            label_dt_list=label_dt_list, 
                                            feature_dt_list=feature_dt_list,  
                                            label_name_format='_index_incr_%id', 
                                            feature_name_format='index_incr_d%i_',
                                            prefix='index',
                                            verbose=verbose)

    index_incr_df.index = index_incr_df['date']
    index_incr_df.index.name = None
    del index_incr_df['date']
    training_set = training_set.join(index_incr_df, on='date')


    ## ================  add fundamental features ================== ## 
    dff = None
    for ds in np.unique(training_set.date.unique()):
        if verbose:
            print("adding fundmentals: %s" % ds)
        sample = get_index_stocks(index_code, date = datetime.datetime.strptime(ds, '%Y%m%d'))
        q = query(
                  valuation.code,
                  valuation.market_cap, 
                  balance.total_assets - balance.total_liability,  # net asset
                  balance.total_assets / balance.total_liability,  # net asset / liability
                  income.net_profit, 
                  income.net_profit + 1, 
                  indicator.inc_revenue_year_on_year, 
                  balance.development_expenditure
            ).filter(
                valuation.code.in_(sample)
            )

        df = get_fundamentals(q, date = None)
        df.index = df.code.values
        del df['code']

        df.columns = ['log_mcap', 'log_NC', 'LEV', 'NI_p', 'NI_n', 'g', 'log_RD']

        df['log_mcap'] = np.log(df['log_mcap'])
        df['log_NC'] = np.log(df['log_NC'])
        df['NI_p'] = np.log(np.abs(df['NI_p']))
        df['NI_n'] = np.log(np.abs(df['NI_n'][df['NI_n'] < 0]))
        df['log_RD'] = np.log(df['log_RD'])

        df = add_industry_feature(df, date=datetime.datetime.strptime(ds, '%Y%m%d'))

        df = df.fillna(0)
        df[df > 10000] = 10000
        df[df < -10000] = -10000

        df['date'] = ds
        df = get_display_name(df, set_index=False)
        df = get_factor(df, factor_name='val_f_')

        if dff is None:
            dff = df.copy()
        else:
            dff = dff.append(df)


    training_set = training_set.merge(dff, 
                             how='left', 
                             left_on=['stock_name', 'date'], 
                             right_on=['display_name', 'date'])

    del training_set['display_name']

    ## ==================== add volume features ===================== ##

    df= get_price(stocks,  
              start_date=start_date, 
              end_date=end_date, 
              frequency=frequency, 
              fields=['volume'], 
              skip_paused=False, 
              fq='pre', 
              count=count)


    df = df['volume'].T

    df = get_display_name(df, set_index=True)       # set stock_name as index
    df.index.name = 'stock_name'

    df = df.fillna(0)
    df = np.log(df + 1) 
    df = df.iloc[:, 1:]
    df = format_df_date(df)   # set column date format as 'yyyymmdd' 


    dff = disassemble_time_series(df=df,
                               start_idx= ts_len,
                               sample_dt = sample_dt,
                               label_dt_list=[],
                               feature_dt_list=feature_dt_list, 
                               label_name_format='_%id_log_volume',
                               feature_name_format = 'log_volume_d%i_',
                               prefix='volume',
                               verbose=verbose)

    training_set = training_set.merge(dff, how = 'left', on=['date', 'stock_name'])

    ## ======= join Beta ====== ## 

    beta_df = None 
    for ds in np.unique(training_set.date.unique()):
        if verbose:
            print("calculating beta: %s" % ds)
        beta = get_index_beta(index_code=index_code, 
                              start_date=None, 
                              end_date=datetime.datetime.strptime(ds, '%Y%m%d'),
                              count=beta_count)
        beta['date'] = ds
        beta['stock_name'] = beta.index
        
        if beta_df is None:
            beta_df = beta
        else:
            beta_df = beta_df.append(beta) 
        
    training_set = training_set.merge(beta_df, how = 'left', on=['date', 'stock_name'])
    
    ## ======= Patch ====== ## 
    fea = ['log_mcap',
           'log_NC', 'LEV', 'NI_p', 'NI_n', 'g', 'log_RD', '801010', '801020',
           '801030', '801040', '801050', '801080', '801110', '801120', '801130',
           '801140', '801150', '801160', '801170', '801180', '801200', '801210',
           '801230', '801710', '801720', '801730', '801740', '801750', '801760',
           '801770', '801780', '801790', '801880', '801890']

    training_set.columns = [s + '_' if s in fea else s for s in training_set.columns]
    
    return training_set



