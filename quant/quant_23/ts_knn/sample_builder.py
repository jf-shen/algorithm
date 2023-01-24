import numpy as np
import pandas as pd
import time
import datetime
import json
import jqdata
from kuanke.user_space_api import *

import types
import functools
from util import Stream, Logger, date2str, str2date, now, get_minute_stat_features, getMinuteStatFeatureThread, timeit, \
    logger


# Start: compatiable preprocess

def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    # g.__kwdefaults__ = f.__kwdefaults__
    return g


_get_price = copy_func(get_price)


def get_price(security,
              start_date=None,
              end_date=None,
              frequency='daily',
              fields=None,
              skip_paused=False,
              fq='pre',
              count=None,
              panel=True):
    panel_result = _get_price(security,
                              start_date=start_date,
                              end_date=end_date,
                              frequency='daily',
                              fields=fields,
                              skip_paused=skip_paused,
                              fq='pre',
                              count=count)

    if panel:
        return panel_result

    df = panel_result.to_frame()
    df['time'] = Stream(df.index).map(lambda p: p[0]).tolist()
    df['code'] = Stream(df.index).map(lambda p: p[1]).tolist()
    df.index = Stream(range(df.shape[0])).tolist()
    return df


# End: compatiable preprocess

class SampleBuilder:
    def __init__(self, index_code="000001.XSHG", pipeline=None, mode='train'):
        # attributes
        self.index_code = index_code
        self.pipeline = pipeline

        self.mode = mode

        self.idx_date_dict = None
        self.date_idx_dict = None

        self.fg = None
        self.logger = logger  # by default, use global logger in default.py

        self.table_name = None
        self.date_list = None

        # data
        self.keys = None
        self.df = None

        # stocks
        self.stock_list = None

        # registered functions
        self.fn_dict = {
            "join_price": self.join_price,
            "join_fundamental": self.join_fundamental,
            "join_industry": self.join_industry,
            "join_index": self.join_index,
            "join_minute_stat": self.join_minute_stat,
            "multi_join_minute_stat": self.multi_join_minute_stat
        }

        self.init_env()

    def head(self, n):
        return self.df.head(n)

    def log_level(self, log_level):
        self.logger.set_log_level(log_level)

    def set_logger(self, logger):
        self.logger = logger

    @timeit(name='init_env')
    def init_env(self):
        """
        make {date:idx} & {idx:date} dict from dplus()
        """
        assert self.index_code is not None
        all_datetime = get_price(security=self.index_code,
                                 fields=[],
                                 start_date=None,
                                 end_date=datetime.datetime.today()).index.tolist()

        all_date = list(map(date2str, all_datetime))

        idx_date_dict = dict(enumerate(all_date))
        date_idx_dict = {v: k for k, v in idx_date_dict.items()}

        self.idx_date_dict = idx_date_dict
        self.date_idx_dict = date_idx_dict

    def dplus(self, ds, dt):
        """
        self.init_env() should be run first
        Args:
            ds: [string] format yyyymmdd
            dt: [int] can be negative
        Return:
            [string] the date of dt transaction days after ds, format: yyyymmdd
        """
        if type(ds) == list:
            return [self.dplus(x, dt) for x in ds]
        else:
            ds_idx = self.date_idx_dict[ds]
            return self.idx_date_dict.get(ds_idx + dt)

    def set_mode(self, mode):
        self.logger.info('switch to %s mode' % mode)
        self.mode = mode

    def filter_null(self):
        self.df = self.df[~self.df.isnull().any(axis=1)]

    def delete(self, cols):
        if type(cols) == list:
            for col in cols:
                del self.df[col]
        else:
            del self.df[cols]
        self.logger.info('delete cols = %s' % cols)

    def set_pipeline(self, pipeline=None):
        if pipeline is None:
            from pipeline import PIPELINE
            pipeline = PIPELINE
        else:
            self.pipeline = pipeline

    def set_keys(self, df):
        self.keys = df[['date', 'code']]

    def set_df(self, df):
        self.df = df
        self.keys = self.set_keys(df)
        return self

    def set_fg(self, fg):
        self.fg = fg

    def load_pipeline(self, json_path):
        with open(json_path, 'r') as f:
            pipeline = json.load(f)

        self.pipeline = pipeline

    def run_pipeline(self):
        for config in self.pipeline.get('sample_builder'):
            fn = self.fn_dict[config['type']]
            params = config.copy()
            params.pop('type')
            fn(**params)

    def build_sample(self,
                     start_date=None,
                     end_date=None,
                     count=None,
                     sample_dt=1,
                     mode='train'):

        start_time = time.time()

        self.generate_keys(start_date=start_date, end_date=end_date, count=count, sample_dt=sample_dt)
        self.set_mode(mode)
        self.run_pipeline()

        self.logger.info("Finish sample building! Time Elasped: %.2f" % (time.time() - start_time))
        start_time = time.time()

        if self.fg is not None:
            self.fg.set_df(self.df)
            self.df = self.fg.generate_features(mode=mode)

        self.logger.info("Finish feature generation! Time Elasped: %.2f" % (time.time() - start_time))

        return self.df

    def append_sample(self,
                      past_df,
                      start_date=None,
                      end_date=None,
                      count=None,
                      sample_dt=1,
                      mode='train'):
        """
        to avoid repetitive computing if past data has already been made
        """

        start_time = time.time()

        self.generate_keys(start_date=start_date, end_date=end_date, count=count, sample_dt=sample_dt)
        self.set_mode(mode)

        key_fields = ['date', 'code']

        df_common = past_df.merge(self.keys, how='inner', on=key_fields)
        key_all = self.keys.merge(past_df[key_fields], how='outer', on=key_fields, indicator=True)
        key_diff = key_all[key_all['_merge'] == 'left_only']
        self.df = key_diff
        self.run_pipeline()

        self.logger.info("Finish additional sample building! Time Elasped: %.2f" % (time.time() - start_time))
        start_time = time.time()

        if self.fg is not None:
            self.fg.set_df(self.df)
            self.df = self.fg.generate_features(mode=mode)

        self.logger.info("Finish feature generation! Time Elasped: %.2f" % (time.time() - start_time))

        self.df = self.df.append(df_common)
        return self.df

    @timeit(name='generate_keys')
    def generate_keys(self,
                      index_code=None,
                      start_date=None,
                      end_date=None,
                      count=None,
                      sample_dt=1
                      ):

        if index_code is None:
            index_code = self.index_code

        assert self.stock_list is not None
        assert index_code is not None

        transaction_date = get_price(security=index_code, fields=[], start_date=start_date, end_date=end_date,
                                     count=count).index.tolist()

        self.date_list = transaction_date[::-sample_dt]  # it seems transaction_date is by default sorted

        keys = []
        for date in self.date_list:
            daily_keys = pd.DataFrame({'code': self.stock_list, 'date': date2str(date)})
            keys.append(daily_keys)

        self.keys = pd.concat(keys)
        self.df = self.keys.copy()
        return self

    def set_stock_list(self, stock_list=None):
        """
        get stock list with [st, st*, delisted, high_limit, low_limit, suspension] removed
        """
        if stock_list is None:
            tf.logging.info("use default stock_list = 'all_stocks'")
            stock_list = get_all_securities(types=['stock']).index.tolist()

        self.stock_list = stock_list
        return self.stock_list

    def join_price(self,
                   dt_list=[-1],
                   field='open',
                   output_format=None,
                   skip_online=None,
                   to_incr=False):
        """
        1. if 'skip_online' is True, the function will not be executed in predict mode
        2. in 'to_incr' mode, initial price/volume is converted to relative increase with repect to current value.
           the original value will not be saved except for 'dt = 0'
        """

        # auto infer skip_online when it is not assigned
        if skip_online is None:
            skip_online = (max(dt_list) > 0)

        def _smart_infer(output_format):
            if output_format is None:
                prefix = '_' if skip_online else ''
                output_format = prefix + field + '_%id'
            if skip_online:
                assert output_format[0] == '_', output_format
            if self.mode == 'predict' and (not skip_online):
                assert max(dt_list) <= 0, dt_list
            return output_format

        output_format = _smart_infer(output_format)

        if self.mode == 'predict' and skip_online:
            self.logger.info('[predict mode]: skip joining %s: dt_list = %s' % (field, dt_list))
            return self

        start_time = time.time()
        self.logger.info('joining %s: dt_list = %s' % (field, dt_list))

        original_dt_list = [dt for dt in dt_list]  # used in log printing
        dt_list = [dt for dt in dt_list]  # copy dt_list so it wouldn't influence the input list
        if to_incr:
            append_zero = (0 not in dt_list) and (output_format % 0 not in self.df.columns.tolist())  # used in the end
            if append_zero:
                dt_list.append(0)

        date_list = self.keys['date'].unique().tolist()

        useful_date_list = []
        for dt in dt_list:
            useful_date_list_for_dt = self.dplus(date_list, dt)
            assert None not in useful_date_list_for_dt, "processing dt = %s, date_list = %s,\
                    useful_date_list_for_dt = %s" % (dt, date_list, useful_date_list_for_dt)
            useful_date_list += useful_date_list_for_dt

        useful_date_list = np.unique(useful_date_list)
        stock_code = self.keys['code'].unique().tolist()

        price_df = get_price(stock_code,
                             start_date=str2date(min(useful_date_list)),
                             end_date=str2date(max(useful_date_list)),
                             frequency='daily',
                             fields=[field],
                             skip_paused=False,
                             fq='pre',
                             count=None,
                             panel=False)

        price_df = price_df[~price_df[field].isnull()]

        price_df['T'] = price_df['time'].map(date2str)
        price_df = price_df[price_df['T'].isin(useful_date_list)]

        for dt in dt_list:
            self.logger.info('append date: %i' % dt, log_level="DEBUG")
            price_df['T%i' % dt] = price_df['T'].map(lambda s: self.dplus(s, -dt))

        for dt in dt_list:
            self.logger.info("join %s: date = %s" % (field, dt), log_level="DEBUG")
            price = price_df[['T%i' % dt, 'code', field]]
            price.columns = ['date', 'code', output_format % dt]
            self.df = self.df.merge(price, how='left', on=['date', 'code'])

        if to_incr:
            for dt in dt_list:
                if dt != 0:
                    self.df[output_format % dt] = self.df[output_format % dt] / self.df[output_format % 0] - 1
            if append_zero:
                del self.df[output_format % 0]

        time_elapsed = time.time() - start_time
        self.logger.info(
            'finish joining %s: dt_list = %s, time elapsed: %.2fs' % (field, original_dt_list, time_elapsed))

        return self

    @timeit(name='join_fundamental')
    def join_fundamental(self, verbose=True):
        """
        Problem: code need to be changed when different fields are queried
        """
        date_list = self.keys['date'].unique().tolist()

        fundamental_df_list = []

        for ds in date_list:
            self.logger.info('joining fundamental: date = %s' % ds, log_level='DEBUG')
            code_list = self.keys[self.keys['date'] == ds]['code'].unique().tolist()

            # https://ycjq.95358.com/data/dict/fundamentals
            q = query(
                valuation.code,
                valuation.market_cap,
                valuation.circulating_market_cap,
                valuation.turnover_ratio,
                valuation.pe_ratio,
                valuation.pe_ratio_lyr,
                valuation.pb_ratio,
                valuation.ps_ratio,
                valuation.pcf_ratio,

                cash_flow.net_finance_cash_flow,
                cash_flow.cash_from_invest,
                cash_flow.goods_sale_and_service_render_cash,

                balance.total_assets,
                balance.total_liability,
                balance.development_expenditure,

                income.operating_revenue,
                income.total_operating_cost,
                income.operating_cost,
                income.financial_expense,
                income.asset_impairment_loss,
                income.investment_income,
                income.operating_profit,
                income.basic_eps,
                income.diluted_eps,

                indicator.operating_profit,
                indicator.roe,
                indicator.roa,
                indicator.net_profit_margin,
                indicator.gross_profit_margin,
                indicator.expense_to_total_revenue,
                indicator.operation_profit_to_total_revenue,
                indicator.ga_expense_to_total_revenue,
                indicator.operating_profit_to_profit,
                indicator.invesment_profit_to_profit,
                indicator.inc_revenue_year_on_year,
                indicator.inc_revenue_annual,
                indicator.inc_operation_profit_year_on_year,
                indicator.inc_operation_profit_annual,
                indicator.inc_net_profit_year_on_year,
                indicator.inc_net_profit_annual
            ).filter(
                valuation.code.in_(code_list)
            )

            query_date = str2date(self.dplus(ds, -1))  # to prevent future information
            fundamental_df = get_fundamentals(q, date=query_date)

            self.logger.info("[join_fundamentals] fundamental_df shape = %s" % str(fundamental_df.shape),
                             log_level='DEBUG')
            fundamental_df['date'] = ds

            fundamental_df_list.append(fundamental_df)

        res_df = pd.concat(fundamental_df_list)
        self.df = self.df.merge(res_df, how='left', on=['date', 'code'])
        return self

    def join_industry(self):
        """
        Problem: haven't taken date into consideration
        """
        self.logger.info("joining industry code ...")

        date_list = self.keys['date'].unique().tolist()
        industry_list = ['801010', '801020', '801030', '801040', '801050', '801080', '801110', '801120', '801130',
                         '801140', '801150', '801160', '801170', '801180', '801200', '801210', '801230', '801710',
                         '801720', '801730', '801740', '801750', '801760', '801770', '801780', '801790', '801880',
                         '801890']
        industry_df = pd.DataFrame({'code': [], 'industry_code': []})
        for industry_code in industry_list:
            stock_code = get_industry_stocks(industry_code, date=str2date(max(date_list)))
            industry_stocks = pd.DataFrame({'code': stock_code, 'industry_code': industry_code})
            industry_df = industry_df.append(industry_stocks)

        assert industry_df.shape[0] == industry_df['code'].unique().shape[0]  # a stock belongs only to one industry

        self.df = self.df.merge(industry_df, how='left', on='code')
        return self

    def join_index(self,
                   dt_list=[-1],
                   field='open',
                   output_format=None,
                   skip_online=None,
                   to_incr=False):
        """
        if skip_online is True, the function will not be executed
        """

        # auto infer skip_online when it is not assigned
        if skip_online is None:
            skip_online = (max(dt_list) > 0)

        def _smart_infer(output_format):
            if output_format is None:
                prefix = '_' if skip_online else ''
                output_format = prefix + 'index_' + field + '_%id'
            if skip_online:
                assert output_format[0] == '_', output_format
            if self.mode == 'predict' and (not skip_online):
                assert max(dt_list) <= 0, dt_list
            return output_format

        output_format = _smart_infer(output_format)

        if self.mode == 'predict' and skip_online:
            self.logger.info('[predict mode]: skip joining index %s: dt_list = %s' % (field, dt_list))
            return self

        start_time = time.time()
        self.logger.info('joining index %s: dt_list = %s' % (field, dt_list))

        original_dt_list = [dt for dt in dt_list]
        dt_list = [dt for dt in dt_list]  # copy dt_list so the original dt_list would not by affected
        if to_incr:
            append_zero = (0 not in dt_list) and (output_format % 0 not in self.df.columns.tolist())  # used in the end
            if append_zero:
                dt_list.append(0)

        index_df = get_price([self.index_code],
                             frequency='daily',
                             fields=[field],
                             end_date=datetime.datetime.today(),
                             skip_paused=False,
                             fq='pre',
                             panel=False)

        index_df = index_df[~index_df[field].isnull()]
        index_df['T'] = index_df['time'].map(date2str)

        for dt in dt_list:
            index_df['T%i' % dt] = index_df['T'].map(lambda s: self.dplus(s, -dt))

        for dt in dt_list:
            index = index_df[['T%i' % dt, field]]
            index.columns = ['date', output_format % dt]
            self.df = self.df.merge(index, how='left', on=['date'])

        if to_incr:
            for dt in dt_list:
                if dt != 0:
                    self.df[output_format % dt] = self.df[output_format % dt] / self.df[output_format % 0] - 1
            if append_zero:
                del self.df[output_format % 0]

        time_elapsed = time.time() - start_time
        self.logger.info(
            'finish joining index %s: dt_list = %s, time elapsed: %.2fs' % (field, original_dt_list, time_elapsed))

        return self

    # ========================== Minute-Level Functions ======================= #

    def join_minute_price(self,
                          field='open',
                          count=60,
                          execute_time='9:35',
                          output_format=None
                          ):

        start_time = time.time()

        if output_format is None:
            output_format = field + '_%dm_'

        date_list = self.keys['date'].unique().tolist()
        stock_code = self.keys['code'].unique().tolist()

        df_list = []

        for date in date_list:
            dt = datetime.datetime.strptime(date + execute_time, '%Y%m%d%H:%M')
            panel = get_price(stock_code,
                              end_date=dt,
                              frequency='minute',
                              skip_paused=False,
                              fields=[field],
                              fq='pre',
                              count=count)
            df = panel[field]
            df = df.iloc[:, :] / df.iloc[0, :]
            df = df.T

            cols = [output_format % (i - count + 1) for i in range(count)]
            df.columns = cols
            df['date'] = date  # date2str(context.current_dt)
            df['code'] = df.index.values
            df = df[['date', 'code'] + cols]

            df_list.append(df)

        min_df = pd.concat(df_list)
        self.df = self.df.merge(min_df, how='left', on=['date', 'code'])

        self.logger.info('[joining_minute_%s] time elapsed = %.2f' % (field, time.time() - start_time))

        return self

    def join_minute_stat(self, field='open', count=60):

        start_time = time.time()
        self.logger.info('start joining minute stats %s: count = %s' % (field, count))

        date_list = self.keys['date'].unique().tolist()
        stock_code = self.keys['code'].unique().tolist()

        min_df = None

        for dt in date_list:
            self.logger.info("start joining minute stats [%s]" % dt, log_level=0)
            dt_start_time = time.time()

            slice_df = get_minute_stat_features(stock_code, str2date(dt), field=field, count=count)
            min_df = slice_df if min_df is None else min_df.append(slice_df)

            self.logger.info("joining minute stats [%s]: time elapsed = %.2f" % (dt, time.time() - dt_start_time),
                             log_level=0)

        min_df = min_df.rename(columns={'mean': 'minute_mean', 'std': 'minute_std'})
        self.df = self.df.merge(min_df, how='left', on=['date', 'code'])

        self.logger.info('finish joining minute stats %s: time elapsed = %.2f' % (field, time.time() - start_time))
        return self

    def multi_join_minute_stat(self, field='open', count=60):
        # multi-thread join (is not suggested)

        start_time = time.time()
        self.logger.info('start multi-joining minute stats %s: count = %s' % (field, count))

        date_list = self.keys['date'].unique().tolist()
        stock_code = self.keys['code'].unique().tolist()

        minute_df_list = []
        threads = []

        for dt in date_list:
            thread = getMinuteStatFeatureThread(minute_df_list, stock_code, dt, field, count)
            thread.start()
            threads.append(thread)

        self.logger.info("thread num = %i: %s" % (len(threads), date_list))

        for t in threads:
            t.join()

        min_df = pd.concat(minute_df_list)
        min_df = min_df.rename(columns={'mean': 'minute_mean', 'std': 'minute_std'})
        self.df = self.df.merge(min_df, how='left', on=['date', 'code'])

        self.logger.info('start multi-joining minute stats %s: time elapsed = %.2f' % (field, time.time() - start_time))
        return self








