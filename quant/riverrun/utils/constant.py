import numpy as np
from

ALL_DATE = np.array(list(map(date2str, get_price(security=index_code,
                             start_date=None,
                             end_date=datetime.datetime.today()).index.tolist())))


