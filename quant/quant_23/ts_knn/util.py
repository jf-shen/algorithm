import datetime
import pandas as pd
import numpy as np
from scipy.stats import norm
import threading
import time

date2str = lambda date: datetime.datetime.strftime(date, '%Y%m%d')
str2date = lambda s: datetime.datetime.strptime(s, '%Y%m%d')
now = lambda: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def select_useful_feature(fea_name):
    return fea_name.endswith('_') or fea_name.start_with('_') or (fea_name in ['code', 'stock_name', 'display_name'])


def get_reward(df, signal, label, a_min, a_max):
    """
    Args:
        df: dataframe containing signal and label
        signal: df columns
        label: df columns
        a_min: signal bucket lower bound
        a_max: signal bucket upper bound
    Return:
        mean(label) for rows satisfying: a_min <= signal < a_max
    """
    a_min = -np.inf if a_min is None else a_min
    a_max = np.inf if a_max is None else a_max
    df['is_candidate'] = (df[signal] >= a_min) & (df[signal] < a_max)
    return df[df['is_candidate']][label].mean()


def group_reward(df, signal, label, bucket_num):
    """
    split data into ${bucket_num} buckets according to signal and
    group rewards in each bucket
    """
    signal_list = np.sort(df[signal])
    x_list = []
    y_list = []
    percentage_list = []
    for i in range(bucket_num):
        idx_min = int(i / float(bucket_num) * len(signal_list))
        idx_max = int((i + 1) / float(bucket_num) * len(signal_list))
        idx_max = min(idx_max, len(signal_list) - 1)

        x_min = signal_list[idx_min]
        x_max = signal_list[idx_max]

        y = get_reward(df, signal=signal, label=label, a_min=x_min, a_max=x_max)
        percentage_list.append(i / float(bucket_num) * 100)
        x_list.append(x_min)
        y_list.append(y)

    return y_list


def group_rewards(df, signals, label, bucket_num, verbose=False):
    res_dict = {}
    for signal in signals:
        if verbose:
            print("grouping rewards by: %s" % signal)
        y_list = group_reward(df, signal, label, bucket_num)
        res_dict[signal] = y_list
    return res_dict


# =============== Inversions testing ================= #

def rank(x):
    """
    get the number of (strictly) ordered-pair of x
    """
    res = 0
    for i in range(len(x)):
        for j in range(i):
            if x[j] < x[i]:
                res += 1
    return res


def rankz(x):
    """
    return normlalized number of ordered-pair which is supposed to have normal distribution
    """
    n = len(x)
    mean = n * (n - 1) / 4
    std = np.sqrt((n - 1) * (n - 2) * (2 * n + 3) / 72)
    z = (rank(x) - mean) / std
    return z


def rankp(x):
    """
    ordered-pair test: return the probability that x has positive trending
    """
    return norm.cdf(rankz(x))


"""
Remark: 
irank(x) is not n(n-1)/2 - rank(x). 
the later didn't take care of the identical values comparison which may cause problem when entries are filled with defalut value.
"""


def irank(x):
    """
    get the number of inversions of x
    """
    res = 0
    for i in range(len(x)):
        for j in range(i):
            if x[j] > x[i]:
                res += 1
    return res


def irankz(x):
    """
    return normlalized number of inversions which is supposed to have normal distribution
    """
    n = len(x)
    mean = n * (n - 1) / 4
    std = np.sqrt((n - 1) * (n - 2) * (2 * n + 3) / 72)
    z = (irank(x) - mean) / std
    return z


def irankp(x):
    """
    inversion test: return the probability that x has negative trending
    """
    return norm.cdf(irankz(x))


# =============== Timer ================= #

def timeit(name=''):
    """
    decorator: get the running time of a function
    """

    def inner(fn):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            output = fn(*args, **kwargs)
            # print("[%s] time elapsed = %.2fs" % (name, time.time() - start_time))
            logger.info("[%s] time elapsed = %.2fs" % (name, time.time() - start_time), log_level=1)
            return output

        return wrapper

    return inner


def trange(iterable, sec=3):
    """
    print progress info every `sec` seconds
    """
    if type(iterable) is int:
        total_num = iterable
        iterable = range(total_num)
    elif isinstance(iterable, pd.DataFrame):
        total_num = iterable.shape[0]
        iterable = iterable.iterrows()
    else:
        # warning: generator can't be processed this way
        total_num = len(iterable)
        iterable = iterable

    update_time = time.time()
    for i, e in enumerate(iterable):
        if time.time() - update_time > sec:
            # print('%i / %i' % (i, total_num))
            logger.info('%i / %i' % (i, total_num))
            update_time = time.time()
        yield e


# =============== Class ================= #
class Logger:
    def __init__(self, log_level=1):
        self.log_level = log_level

        # log level mapping
        self.log_level_dict = {
            "DEBUG": 0, "DEFAULT": 1, "IMPORTANT": 2, "COERSIVE": 3,
            "debug": 0, "default": 1, "important": 2, "coersive": 3
        }

    def set_log_level(self, log_level):
        if type(log_level) == str:
            log_level = self.log_level_dict[log_level]
        self.log_level = log_level

    def info(self, msg, log_level=1):
        if type(log_level) == str:
            log_level = self.log_level_dict[log_level]
        if log_level >= self.log_level:
            print(msg)


logger = Logger(log_level=1)


class Stream:
    def __init__(self, obj):
        self.iterable = obj

        if type(obj) == int:
            self.iterable = range(obj)

            # transform

    def map(self, fn):
        self.iterable = map(fn, self.iterable)
        return self

    def filter(self, fn):
        self.iterable = filter(fn, self.iterable)
        return self

    def unique(self):
        self.iterable = list(set(self.iterable))
        return self

    # groupby
    def groupby(self, key_fn=None):
        if key_fn is not None:
            self.iterable = map(lambda row: (key_fn(row), row), self.iterable)

        collect_dict = dict()
        for k, v in self.iterable:
            if k in collect_dict:
                collect_dict[k].append(v)
            else:
                collect_dict[k] = [v]
        self.iterable = collect_dict.items()
        return self

    def agg(self, agg_fn):
        res_dict = dict()
        for k, v_list in self.iterable:
            res_dict[k] = reduce(agg_fn, v_list)
        self.iterable = res_dict.items()
        return self

    def keys(self):
        self.iterable = map(lambda kv: kv[0], self.iterable)
        return self

    def values(self):
        self.iterable = map(lambda kv: kv[1], self.iterable)
        return self

    # execute
    def len(self):
        return len(list(self.iterable))

    def sum(self):
        return sum(list(self.iterable))

    def max(self):
        return max(list(self.iterable))

    def min(self):
        return min(list(self.iterable))

    def tolist(self):
        return list(self.iterable)

    def to_dict(self):
        return dict(self.iterable)

    def to_enum(self):
        return dict(enumerate(self.iterable))

    def head(self, num=3):
        lst = []
        for it in self.iterable:
            lst.append(it)
            if len(lst) > num:
                break
        return lst


# =============== DataFrame Functions ================= #
def filter_abnormal(df):
    # filter abnormal rows in df
    return df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]


def get_null(df):
    # get_null chunk from df
    null_col = df.isnull().any(axis=0).index.tolist()
    null_row = df.isnull().any(axis=1)
    return df[null_col][null_row]


def corr(df, signal='pred', label='_price_incr_7d'):
    # corr of signal and label
    not_null_idx = ~(df[signal].isnull() | df[label].isnull())
    return np.corrcoef(df[signal][not_null_idx], df[label][not_null_idx])[0, 1]


def topk(df, k=10, signal='pred', label='_price_incr_7d', return_incr=True):
    # topk of signal reward mean VS all reward mean
    not_null_idx = ~(df[signal].isnull() | df[label].isnull())
    df = df[not_null_idx].sort_index(by=signal, ascending=False)
    if return_incr:
        return df.iloc[:k, :][label].mean() - df[label].mean()
    else:
        return df.iloc[:k, :][label].mean()


# =============== Involving JQ-data ================= #

def diff_df(df, dt=1):
    dff = df.copy()
    for i in range(df.shape[1] - dt):
        dff.iloc[:, i + dt] = (df.iloc[:, i + dt] - df.iloc[:, i]) / df.iloc[:, i]
    dff = dff.iloc[:, dt:]  # remove init prices
    return dff


def get_index_beta(index_code,
                   end_date,
                   count,
                   frequency='daily',
                   price_field='open'):
    stocks = get_index_stocks(index_code, date=end_date)
    df = get_price(stocks,
                   end_date=end_date,
                   count=count,
                   frequency=frequency,
                   fields=[price_field],
                   skip_paused=False,
                   fq='pre')

    df = df[price_field].T

    df = diff_df(df, dt=1)  # price -> increase
    df = df.fillna(0)

    df.columns = list(map(date2str, df.columns))
    # display(df.head(3))

    index_df = get_price(index_code,
                         end_date=end_date,
                         count=count,
                         frequency=frequency,
                         fields=[price_field],
                         skip_paused=False,
                         fq='pre')

    index_df = diff_df(index_df.T, dt=1)  # price -> increase
    index_ts = index_df.values[0]

    beta_dict = {}
    for i in range(df.shape[0]):
        stock_ts = df.iloc[i, :]
        idx = df.index[i]
        beta_dict[idx] = np.corrcoef(stock_ts, index_ts)[0, 1]

    beta_df = pd.DataFrame.from_dict(beta_dict, orient='index')

    beta_df.columns = ['beta']
    beta_df = beta_df.sort_index(by='beta')
    return beta_df


# =============== Minute Data ================= #


def get_minute_stat_features(stock_code, current_dt, field='open', count=60):
    """
    Arg:
        stock_code: stock_code list
        current_dt: a timestamp
        field: field of minite-level K-bar to be taken
        count: the number of minite K-bar before current_dt
    Return:
        res_df: statistics of minutes time theory
    """
    panel = get_price(stock_code,
                      end_date=current_dt,
                      frequency='minute',
                      skip_paused=False,
                      fields=[field],
                      fq='pre',
                      count=count)

    df = panel['open'].T
    diff = np.log(df.values[:, 1:] / df.values[:, :-1])
    res_dict = {}

    res_dict['mean'] = np.mean(diff, axis=1)
    res_dict['std'] = np.std(diff, axis=1)

    res_df = pd.DataFrame.from_dict(res_dict, orient='columns')
    res_df['code'] = df.index
    res_df['date'] = date2str(current_dt)

    res_df = res_df[['code', 'date'] + res_dict.keys()]

    return res_df


threadLock = threading.Lock()


class getMinuteStatFeatureThread(threading.Thread):
    def __init__(self, minute_df_list, stock_code, current_dt, field='open', count=60, logger=None):
        threading.Thread.__init__(self)
        self.stock_code = stock_code
        self.current_dt = current_dt
        self.field = field
        self.count = count
        self.minute_df_list = minute_df_list
        self.logger = logger if logger is not None else Logger(log_level=1)

    def run(self):
        start_time = time.time()
        slice_df = get_minute_stat_features(self.stock_code, str2date(self.current_dt), field=self.field,
                                            count=self.count)

        threadLock.acquire()
        self.minute_df_list.append(slice_df)
        threadLock.release()

        self.logger.info(
            '[%s] finish thread job: thread time elapse = %.2f' % (self.current_dt, time.time() - start_time))


