import datetime
import numpy as np
from jqdatasdk import get_price

def date2str(date):
    return datetime.datetime.strftime(date, '%Y%m%d')


def str2date(s):
    return datetime.datetime.strptime(s, '%Y%m%d')

ALL_DATE = np.array(list(map(date2str, get_price(
        security=index_code,
        start_date=None,
        end_date=datetime.datetime.today()).index.tolist())))


def dplus(ds, dt):
    idx = np.where(ALL_DATE == ds)[0][0]
    return ALL_DATE[idx + dt]