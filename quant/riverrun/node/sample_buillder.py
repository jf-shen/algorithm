import numpy as np
import pandas as pd
import datetime
from quant import *
from dataset import DataSet
import sklearn
from sklearn import linear_model


class SampleBuilder:
    def __init__(self,
                 index_code='000001.XSHG',
                 start_date=None,
                 end_date=None,
                 count=None,
                 sample_dt=1,
                 label_dt_list=[1, 3, 7, 14, 28, 90],
                 feature_dt_list=[0, -1, -2, -3, -4, -5, -6, -7]):

        self.index_code = index_code
        self.start_date = start_date
        self.end_date = end_date
        self.count = count
        self.sample_dt = sample_dt
        self.label_dt_list = label_dt_list
        self.feature_dt_list = feature_dt_list

        self.process = []  # [{'name': {'func':func, 'param':param}}]

    def dplus(ds, dt):
        idx = np.where(ALL_DATE == ds)[0][0]

    return ALL_DATE[idx + dt]

    def generate_key(self):
        start_date = self.date_date
        end_date = self.end_date
        count = self.count
        sample_dt = self.sample_dt

        transaction_date = get_price(security=index_code,
                                     start_date=start_date,
                                     end_date=end_date,
                                     count=count).index.tolist()

        sample_date = []
        for i in range(len(transaction_date)):
            if (len(transaction_date) - i - 1) % sample_dt == 0:
                sample_date.append(transaction_date[i])

        df = pd.DataFrame({'code': [], 'date': []})
        for date in sample_date:
            stock_code = get_index_stocks(index_code, date=date)
            df_tmp = pd.DataFrame({'code': stock_code, 'date': date2str(date)})
            df = df.append(df_tmp)

        print('finish key generation')
        return df

    def join_price(df, dt_list=[-1], field='open', output_format='price_%id', verbose=True):
        print('joining: %s, dt_list = %s' % (field, dt_list))

        date_list = df['date'].unique()
        start_date = dplus(min(date_list), min(dt_list))
        end_date = dplus(max(date_list), max(dt_list))
        stock_code = df['code'].unique().tolist()
        price_df = get_price(stock_code,
                             start_date=str2date(start_date),
                             end_date=str2date(end_date),
                             frequency=frequency,
                             fields=[field],
                             skip_paused=False,
                             fq='pre',
                             count=None,
                             panel=False)

        price_df['T'] = price_df['time'].map(date2str)
        for dt in dt_list:
            if verbose:
                print('append date: %i' % dt)
            price_df['T%i' % dt] = price_df['T'].map(lambda s: dplus(s, -dt))

        for dt in dt_list:
            if verbose:
                print("join %s: date = %s" % (field, dt))
            price = price_df[['T%i' % dt, 'code', field]]
            price.columns = ['date', 'code', output_format % dt]
            df = df.merge(price, how='left', on=['date', 'code'])

        return df

    def join_fundamental(df, verbose=True):
        date_list = df['date'].unique()
        res_df = None
        for ds in date_list:
            if verbose:
                print('joining fundamental: date = %s' % ds)
            sample = get_index_stocks(index_code, date=str2date(ds))
            q = query(
                valuation.code,
                valuation.market_cap,
                balance.total_assets,
                balance.total_liability,
                indicator.inc_revenue_year_on_year,
                balance.development_expenditure
            ).filter(
                valuation.code.in_(sample)
            )

            fundamental_df = get_fundamentals(q, date=str2date(ds))
            fundamental_df['date'] = ds
            if res_df is None:
                res_df = fundamental_df
            else:
                res_df = res_df.append(fundamental_df)

        df = df.merge(res_df, how='left', on=['date', 'code'])
        return df

    def join_industry(df):
        # don't take date into consideration for now
        date_list = df['date'].unique()
        industry_list = ['801010', '801020', '801030', '801040', '801050', '801080', '801110', '801120', '801130',
                         '801140', '801150', '801160', '801170', '801180', '801200', '801210', '801230', '801710',
                         '801720', '801730', '801740', '801750', '801760', '801770', '801780', '801790', '801880',
                         '801890']
        industry_df = None
        for industry_code in industry_list:
            stock_code = get_industry_stocks(industry_code, date=str2date(max(date_list)))
            industry_stocks = pd.DataFrame({'code': stock_code, 'industry_code': industry_code})
            if industry_df is None:
                industry_df = industry_stocks
            else:
                industry_df = industry_df.append(industry_stocks)

        assert industry_df.shape[0] == industry_df['code'].unique().shape[0]  # a stock belongs only to one industry
        df = df.merge(industry_df, how='left', on='code')
        return df
