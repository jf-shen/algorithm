import datetime

date2str = lambda date: datetime.datetime.strftime(date, '%Y%m%d')
str2date = lambda s: datetime.datetime.strptime(s, '%Y%m%d')
now = lambda: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class Stream():
    def __init__(self, obj):
        self.iterable = obj

    # transform
    def map(self, fn):
        self.iterable = map(fn, self.iterable)
        return self

    def filter(self, fn):
        self.iterable = filter(fn, self.iterable)
        return self

    # execute
    def len(self):
        return len(list(self.iterable))

    def sum(self):
        return sum(list(self.iterable))

    def tolist(self):
        return list(self.iterable)

    def to_dict(self):
        return dict(self.iterable)


## compute price (relative) incr
def diff_df(df, dt=1, remove_init_price=False):
    dff = df.copy()
    for i in range(df.shape[1] - dt):
        dff.iloc[:, i + dt] = (df.iloc[:, i + dt] - df.iloc[:, i]) / df.iloc[:, i]

    if remove_init_price:
        dff = dff.iloc[:, dt:]

    return dff


## get_null chunk from df
def get_null(df):
    null_col = df.isnull().any(axis=0).index.tolist()
    null_row = df.isnull().any(axis=1)
    return df[null_col][null_row]





