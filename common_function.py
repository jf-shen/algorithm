import numpy as np 
import datetime 


# ==============================  日期函数  ============================== # 

to_date = lambda s: datetime.datetime.strptime(s, '%Y-%m-%d')

"""
求date1与date2间隔天数, date1、date2的顺序无所谓
@date1 格式: '%Y-%m-%d'
@date2 格式：'%Y-%m-%d'
"""

def datediff(date1, date2):
    timedelta = to_date(date1) - to_date(date2)
    return abs(timedelta.days)


# ==============================  函数式编程  ============================== # 

# filter dict key
def filter_key(func, dict_instance):
    iterable_object = filter(lambda x: func(x[0]), dict_instance.items())
    return dict(iterable_object)

# filter dict value
def filter_value(func, dict_instance):
    iterable_object = filter(lambda x: func(x[1]), dict_instance.items())
    return dict(iterable_object)

