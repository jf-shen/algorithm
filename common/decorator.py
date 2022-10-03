import time


def timeit(name=''):
    def inner(fn):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            fn(*args, **kwargs)
            print("[%s] %.3f" % (name, time.time() - start_time))

        return wrapper

    return inner


if __name__ == '__main__':
    @timeit(name="my_func")
    def my_func(sec):
        print('my_func is called')
        time.sleep(sec)

    my_func(1)
