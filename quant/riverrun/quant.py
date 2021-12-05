import numpy as np 
import pandas as pd
import sklearn


INDUSTRY_LIST =  ['801010', '801020', '801030', '801040', '801050', '801080', '801110', '801120', '801130', '801140', '801150', '801160', '801170', '801180', '801200', '801210', '801230', '801710', '801720', '801730', '801740', '801750', '801760', '801770', '801780', '801790', '801880','801890']

# ================== basic functions  ====================  # 
def corr(x, y):
    return (np.mean(x*y) - np.mean(x) * np.mean(y))   / np.std(x) / np.std(y)


def get_display_name(df, set_index = False):
    df_name = get_all_securities()[['display_name']]
    df = df.join(df_name)

    if set_index:
        df.index = df['display_name']
        del df['display_name']
    else:
        df = df[df.columns[-1:].tolist() + df.columns[:-1].tolist()] 

    return df

def diff_df(df, dt = 1, remove_init_price = False):
    """
    dX(t) = [X(t+dt) - X(t)] / X(t) 
    """
    dff = df.copy()
    for i in range(df.shape[1] - dt):
        dff.iloc[:,i+dt] = (df.iloc[:, i+dt] - df.iloc[:, i]) / df.iloc[:, i]
    
    if remove_init_price:
        dff = dff.iloc[:, dt:]
    
    return dff 


def format_df_date(df, date_format = '%Y%m%d'):
    if type(df.columns[0]) == pd.tslib.Timestamp:
        df.columns =  list(map(lambda s: s.strftime(date_format), df.columns))
    if type(df.index[0]) == pd.tslib.Timestamp:
        df.index =  list(map(lambda s: s.strftime(date_format), df.index))
    return df


def accum_incr(x):
    """
    sum up increments
    """
    return np.exp(np.sum(np.log(1 + x))) - 1 


# ================== feature functions ====================  # 

def disassemble_time_series(df, 
                            start_idx, 
                            sample_dt, 
                            label_dt_list, 
                            feature_dt_list, 
                            label_name_format='_%i', 
                            feature_name_format = '%i_',
                            prefix='',
                            verbose=False):
    """
    disassemble the time series dataframe to a dataframe of training sample indexed by (stock, date)
    Args:
        @df: time series dataframe, each row a time series for a stock, with column format "yyyymmdd"
        @start_idx: sampling begins at start_idx's column 
        @sample_dt: the time gap between sampling. 
        @label_dt_list: 
            cumulative dt-day increases will be calculated as labels.
            valid only when df is price increse time series. 
            example: label_dt_list=[1,3,7], then 1-day, 3-day and 7-day cumulative increases will be calculated as lables.
        @feature_dt_list: 
            day-dt's time series value is reserved for feature.
            example: feature_dt_list=[1,2,3], then day-1, day-2, day-3 time series value will be put into features. 
        @label_name_format: a template for returned label columns name, example: '_index_incr_%id'
        @feature_name_format: a template for returned feature columns name, example: 'index_incr_d%i'
        @prefix: prefix of log
        @verbose: print log if verbose is true.
    Returns:
        a dataframe, each row as a training sample.  
    """
    
    res_dict = dict()
    
    label_key_set = map(lambda dt: (label_name_format % dt), label_dt_list)
    for k in label_key_set:
        res_dict[k] = []
        
    feature_key_set = map(lambda dt: (feature_name_format % dt), feature_dt_list)
    for k in feature_key_set:
        res_dict[k] = [] 
    
    res_dict['date'] = []
    
    index_name = df.index.name if df.index.name is not None else "index"
    res_dict[index_name] = []
    
    assert max(feature_dt_list) <= start_idx,  "feature length: %d, start index: %d" % (max(feature_dt_list), start_idx)
    
    sample_idx_list = list(range(start_idx, df.shape[1] - max(label_dt_list + [0]), sample_dt))
    
    for sample_idx in sample_idx_list:
        if verbose:
            print("[%s] making sample: %s" % (prefix, df.columns[sample_idx]))
        
        # cumulative increase as labels: 
        for label_dt in label_dt_list:
            res_dict[label_name_format % label_dt] += \
                df.iloc[:, sample_idx:(sample_idx + label_dt)].apply(accum_incr, axis=1).tolist()
        
        # single-day increase as features: 
        for feature_dt in feature_dt_list:
            res_dict[feature_name_format % feature_dt] += df.iloc[:, sample_idx - feature_dt].tolist()
                
        res_dict['date'] += [df.columns[sample_idx]] * df.shape[0]
        res_dict[index_name] += df.index.tolist()
    
    res_df = pd.DataFrame(res_dict)
    
    # reorder the columns 
    col_order = [index_name, 'date'] + [(label_name_format % dt) for dt in np.sort(label_dt_list)] + \
                [(feature_name_format % dt) for dt in np.sort(feature_dt_list)]
    res_df = res_df[col_order]
    
    return res_df


def add_industry_feature(df, code_name=None, date=None):
    industry_list = ['801010', '801020', '801030', '801040', '801050', '801080', '801110', '801120', '801130', '801140', '801150', '801160', '801170', '801180', '801200', '801210', '801230', '801710', '801720', '801730', '801740', '801750', '801760', '801770', '801780', '801790', '801880','801890']
    
    code_col = df.index if code_name is None else df[code_name].tolist() 
    
    for industry_code in industry_list:
        industry_stocks = get_industry_stocks(industry_code, date=date)
        cate_df = pd.DataFrame({'index': industry_stocks})
        cate_df[industry_code] = 1
        cate_df.index = cate_df['index']
        del cate_df['index']
        
        df = df.join(cate_df, on=code_name)
        df[industry_code].fillna(0, inplace=True)
        
    return df



def get_index_beta(index_code, 
                   start_date=None, 
                   end_date=None, 
                   count=None, 
                   frequency='daily',
                   stock_name_as_index=True):
    """
    calculate all stocks' beta for stocks in index_code, based on data from start_date to end_date
    Args:
        index_code: index code for beta
        start_date: start date of time series used to calculate beta 
        end_date: end date of time series used to calculate beta
        count: is None if start_date and end_date is specified 
        frequency: bar frequency, default to be 'daily' 
        stock_name_as_index: use stock_name as index if is True, otherwise, use stock_code. 
    Return:
        beta_df: {stock_name:[...], beta: [...]}
    """
    
    assert (start_date is None) + (end_date is None) + (count is None) == 1, \
        "start_date = %s, end_date = %s, count = %s" % (start_date, end_date, count)

    stocks = get_index_stocks(index_code, date = end_date) 
    df= get_price(stocks,  
              start_date=start_date, 
              end_date=end_date, 
              frequency=frequency, 
              fields=['close'], 
              skip_paused=False, 
              fq='pre', 
              count=count)
    df = df['close'].T

    if stock_name_as_index:
        df = get_display_name(df, set_index = True)   # set stock name as index

    df = diff_df(df, dt = 1, remove_init_price=True)  # price -> increase
    df = df.fillna(0)
    df = format_df_date(df)  # column format: yyyymmdd 

    index_df = get_price(index_code, 
              start_date=start_date, 
              end_date=end_date, 
              frequency=frequency, 
              fields=['close'], 
              skip_paused=False, 
              fq='pre', 
              count=count)

    index_df = diff_df(index_df.T, dt = 1, remove_init_price=True)  # price -> increase
    index_ts = index_df.values[0]

    beta_dict = {}
    for i in range(df.shape[0]):
        stock_ts = df.iloc[i, :]
        idx = df.index[i]
        beta_dict[idx] = corr(stock_ts, index_ts)

    beta_df = pd.DataFrame.from_dict(beta_dict, orient='index')

    beta_df.columns = ['beta']
    beta_df = beta_df.sort_values(by = 'beta')
    return beta_df
  
# ==================== factor function ====================  #  
def get_factor(df, factor_name='factor'):
    """
    get the fundamental factor with linear regression (To be improved ...)
    """
    label = ['log_mcap']
    features = ['log_NC', 'LEV', 'NI_p', 'NI_n', 'g', 'log_RD'] + INDUSTRY_LIST
    Y = df[label]
    X = df[features]
    model = sklearn.linear_model.ElasticNet(alpha=1, l1_ratio=0.5)
    model.fit(X,Y)

    factor = Y - pd.DataFrame(model.predict(X), index = Y.index, columns = ['log_mcap'])    ## log(real val) - log(pred val)
    # display(factor.head(5))
    factor = factor.sort_values(by = 'log_mcap')
    
    df[factor_name] = factor['log_mcap']

    return df


# ================== data analysis function ====================  # 
def normalize(df, columns=None, eps = 1e-8, return_param=False):
    param_dict = {
        'name': [],
        'mean': [],
        'std': []
    } 
    columns = columns if columns is not None else df.columns
    for column in columns:
        param_dict['name'].append(column)
        param_dict['mean'].append(df[column].mean())
        param_dict['std'].append(df[column].std())
        
        df[column] = (df[column] - df[column].mean()) / (df[column].std() + eps)
    
    param_df = pd.DataFrame(param_dict)
    param_df = param_df.set_index('name')
    param_df.index.name = None 
    
    if return_param:
        return df, param_df
    else:
        return df

def lin_reg(df, label=None, features=[], l2=1e-4, return_df=True): 
    X = np.array(df[features].values)
    Y = np.array(df[label].values) 
    
    N, d = X.shape
    
    beta = np.linalg.inv(X.T.dot(X) + l2 * np.eye(d)).dot(X.T).dot(Y)
    coeff_dict = dict(zip(features, beta))
    
    if return_df:
        coeff_df = pd.DataFrame.from_dict(coeff_dict, orient='index')
        coeff_df.columns = ['coeff']
        coeff_df = coeff_df.reindex(index = features)
        return coeff_df
    
    return coeff_dict
    
