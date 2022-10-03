import types
import functools
from functools import reduce


# ====================== Deep Copy ========================== #
def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


# ====================== Stream ========================== #
class Stream:
    def __init__(self, obj):
        self.iterable = obj

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

    def quantile(self, p):
        """
        need self.iterable to be sortable
        """
        lst = sorted(self.iterable)
        return lst[int[len(lst) * p]]


# ====================== 函数复合 ========================== #
def compose(*func):
    """
    compose a list of functions
    Args:
        func: a tuple of functions: (f1, f2, ..., fn)
    Return:
        the composed function: f1.f2.f3. ... .fn
    Example:
        fn = compose(lambda x: x+1, lambda x: x*2), fn(3) = 7
    """
    def apply_all(x, *func):
        y = x
        for f in func[::-1]:
            y = f(y)
        return y
    fn = lambda x: apply_all(x, *func)
    return fn


def pipe(*func):
    """
    pipe a list of functions
    Args:
        @param func: a tuple of functions: (f1, f2, ..., fn)
    Return:
        the composed function: fn. ... f2.f1
    Example:
        fn = compose(lambda x: x+1, lambda x: x*2), fn(3) = 8
    """
    def apply_all(x, *func):
        y = x
        for f in func:
            y = f(y)
        return y

    fn = lambda x: apply_all(x, *func)
    return fn


# ====================== arg min ======================= #

more_than = lambda x, y: x > y
less_than = lambda x, y: x < y


def argopt_(domain, fn, better_than):
    """
    Args:
        domain: an iterable object
        fn: func(domain -> range)
        better_than: better_than(y1, y2) = True if (y1 is optimal than y2) for (y1, y2 in range)
    Return:
        the first optimal element in domain
    """
    opt_x = None
    opt_y = None
    for x in domain:
        y = fn(x)
        if (opt_x is None) or better_than(y, opt_y):
            opt_x = x
            opt_y = y
    return opt_x


def argopt(obj, fn, better_than):
    """
    Args:
        @obj: an object that auto domain inference is performed on
        @fn: a fucntion from domain to range
        @better_than: better_than(y1, y2) = True if y1 is optimal than y2 (for y1, y2 in range)
    Return:
        the first optimal element in domain
    """
    if type(obj) in [list, np.ndarray]:
        domain = list(range(len(obj)))
        if fn is None:
            fn = lambda i: obj[i]
    elif type(obj) in [dict]:
        domain = obj.keys()
        if fn is None:
            fn = lambda k: obj[k]
    else:
        domain = obj
        fn = fn
    return argopt_(domain, fn, better_than)


def argmin(obj, fn=None):
    return argopt(obj=obj, fn=fn, better_than=less_than)


def argmax(obj, fn=None):
    return argopt(obj=obj, fn=fn, better_than=more_than)

