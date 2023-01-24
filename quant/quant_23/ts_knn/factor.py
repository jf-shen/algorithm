import numpy as np
import pandas as pd
import time
import datetime
from util import *
from sample_builder import SampleBuilder
from jqdata import *
from kuanke.user_space_api import *
from six import StringIO
import os


class KnnFactor:
    def __init__(self, context, g):
        self.context = context
        self.freq = g.freq

        self.pattern_len = 7
        self.knn_perception_threshold = 1.0 / 3
        self.date_weighting_fn = np.sqrt
        self.stock_weighting_fn = np.sqrt

        self.data_path = 'data/ts_knn/'
        self.recent_factor_return_win_size = 14  # 98/freq
        self.stock_list = select_stocks(context)

        self.candidate_df = None
        self.pattern_df = None
        self.res_df = None

    def set_data_path(self, data_path):
        self.data_path = data_path

    def run(self):
        pattern_df = self.pattern_scan()
        pattern_df = filter_abnormal(pattern_df)

        candidate_df = self.get_candidate_df()
        res_df = self.pattern_match(candidate_df, pattern_df)

        candidate_df = candidate_df.merge(res_df, on=['code', 'date'])
        candidate_df['avg_f_'] = candidate_df[['open_f_', 'close_f_', 'high_f_', 'low_f_', 'volume_f_']].mean(axis=1)
        candidate_df = candidate_df.sort_index(by='avg_f_', ascending=False)

        self.candidate_df = candidate_df
        self.pattern_df = pattern_df
        self.res_df = res_df
        return candidate_df

    @timeit(name='pattern_match')
    def pattern_match(self, candidate_df, pattern_df):
        res_dict = {
            "code": [],
            "date": [],
            "open_f_": [],
            "close_f_": [],
            "high_f_": [],
            "low_f_": [],
            "volume_f_": []
        }

        def rbf(x, sigma=1):
            x = np.array(x)
            return np.exp(- x ** 2 / (2 * sigma ** 2))

        def get_val(row, field, df):
            cols = Stream(self.pattern_len).map(lambda i: '%s_%id_' % (field, (-i - 1))).tolist()
            my_x = row[cols]

            dis = df[cols].apply(lambda x: x * x).sum(axis=1) - 2 * df[cols].dot(my_x) + np.sum(my_x * my_x)
            dis = np.sqrt(np.array(dis, dtype=float))
            pattern_sigma = np.sqrt(df[cols].var().sum())  # center sigma

            dis = rbf(dis, sigma=pattern_sigma * self.knn_perception_threshold)
            # val = (dis * df['weight']).sum() / df['weight'].sum()  # weighted sum
            # incr = (df['_open_7d'] * dis * df['weight']).sum() / (dis * df['weight']).sum()
            val = (dis * df['weight'] * df['_open_%id' % self.freq]).sum() / df['weight'].sum()
            return val

        for row in trange(candidate_df):
            row = row[1]
            res_dict['code'].append(row['code'])
            res_dict['date'].append(row['date'])

            df = pattern_df[pattern_df['date'] < row['date']]

            for field in ["open", "close", "high", "low", "volume"]:
                dis = get_val(row, field, df)
                res_dict[field + '_f_'].append(dis)

        res_df = pd.DataFrame(res_dict)
        return res_df

    @timeit(name='get_candidate_df')
    def get_candidate_df(self):
        sb2 = SampleBuilder(index_code='000001.XSHG')
        sb2.set_stock_list(self.stock_list)

        sb2.generate_keys(
            index_code='000001.XSHG',
            end_date=self.context.current_dt,
            count=1,
            sample_dt=1
        )

        dt_list = Stream(self.pattern_len).map(lambda x: -x - 1).tolist()
        for field in ["open", "close", "high", "low", "volume"]:
            sb2.filter_null()
            sb2.join_price(
                dt_list=[dt for dt in dt_list],
                output_format=field + "_%id_",
                field=field,
                skip_online=False,
                to_incr=True
            )
        return sb2.df

    @timeit(name='pattern_scan')
    def pattern_scan(self):
        # end_date = self.context.current_dt - datetime.timedelta(days=self.freq+1)
        sb = SampleBuilder(index_code='000001.XSHG')
        sb.set_stock_list(self.stock_list)

        end_date = sb.dplus(date2str(self.context.current_dt), -self.freq)  # -(self.freq + 1))
        end_date = str2date(end_date) + datetime.timedelta(hours=9.5)

        sb.generate_keys(
            index_code='000001.XSHG',
            end_date=end_date,
            count=200 * 3,
            sample_dt=1
        )

        sb.join_price(
            dt_list=[self.freq],
            output_format="_open_%id",
            field="open",
            skip_online=False,
            to_incr=True
        )

        sb.df = sb.df[sb.df['_open_%id' % self.freq] > 0.1]
        sb.keys = sb.df[['code', 'date']]
        sb.df = sb.keys

        dt_list = Stream(self.freq).map(lambda x: x + 1).tolist()
        sb.join_price(
            dt_list=[dt for dt in dt_list],
            output_format="_open_%id",
            field="open",
            skip_online=False,
            to_incr=True
        )

        if self.freq > 3:
            sb.df['rankp'] = sb.df[Stream(range(self.freq)).map(lambda x: '_open_%id' % (x + 1)).tolist()].apply(rankp,
                                                                                                                 axis=1)
            sb.df = sb.df[sb.df['rankp'] > 0.99]

        dt_list = Stream(self.pattern_len).map(lambda x: -x - 1).tolist()
        for field in ["open", "close", "high", "low", "volume"]:
            sb.filter_null()
            sb.join_price(
                dt_list=dt_list,
                output_format=field + "_%id_",
                field=field,
                skip_online=False,
                to_incr=True
            )

        def unique(x, to_df=True):
            pair = np.unique(x, return_counts=True)
            k = pair[0]
            v = pair[1]
            if not to_df:
                return dict(zip(k, v))
            df_kv = pd.DataFrame.from_dict({'value': k, 'count': v})
            return df_kv[['value', 'count']].sort_index(by='count', ascending=False)

        date_num_dict = unique(sb.df['date'].values, to_df=False)
        code_num_dict = unique(sb.df['code'].values, to_df=False)

        sb.df['date_num'] = sb.df['date'].apply(lambda s: date_num_dict.get(s, 0))
        sb.df['code_num'] = sb.df['code'].apply(lambda s: code_num_dict.get(s, 0))

        sb.df['weight'] = 1 / self.date_weighting_fn(sb.df['date_num']) / self.stock_weighting_fn(sb.df['code_num'])
        return sb.df

    def fill_last_reward(self):
        file_info_path = self.data_path + 'info.txt'  # e.g.'data/ts_knn/info'
        label = '_open_%id' % self.freq

        #         if not os.path.exists(file_info_path):
        #             print("file_info_path does not exist: " + file_info_path)
        #             return

        try:
            file_list = read_file(file_info_path)
        except:
            logger.info("file_info_path does not exist: " + file_info_path, 2)
            return

        file_list = file_list.split('\n')
        if len(file_list) == 0:
            logger.info("file_info_path empty!", log_level=2)
            return None

        most_recent_file_path = max(file_list)

        def read_csv(file_name):
            body = read_file(file_name)
            return pd.read_csv(StringIO(body), index_col=0)

        df = read_csv(most_recent_file_path)
        if not (df[label].max() == 0 and df[label].min() == 0):
            logger.info("last df has already been filled with reward!", 2)
            return None

        df['date'] = df['date'].astype(str)
        end_date = df['date'].unique()[0]

        sb2 = SampleBuilder(index_code='000001.XSHG')
        sb2.set_stock_list(self.stock_list)
        sb2.generate_keys(
            index_code='000001.XSHG',
            end_date=str2date(end_date),
            count=1,
            sample_dt=1
        )
        sb2.join_price(
            dt_list=[self.freq],
            output_format="_open_%id",
            field="open",
            skip_online=True,
            to_incr=True
        )
        reward_df = sb2.df[['date', 'code', label]]

        if label in df.columns.tolist():
            del df[label]

        df = df.merge(reward_df, on=['date', 'code'])
        info_df = df.sort_index(by='avg_f_', ascending=False).head(5)[['code', 'avg_f_', label]]

        logger.info("[last reward]:", log_level=2)
        logger.info(info_df, log_level=2)

        write_file(most_recent_file_path, df.to_csv(), append=False)

    def write_data(self):
        context = self.context

        file_path = self.data_path + date2str(context.current_dt) + '.csv'  # 'data/ts_knn/20221030.csv'
        file_info_path = self.data_path + 'info.txt'  # 'data/ts_knn/info'
        write_file(file_info_path, file_path + '\n', append=True)

        record_df = self.candidate_df[['date', 'code',
                                       # '_open_%id' % self.freq, '_index_open_%id' % self.freq,
                                       'open_f_', 'close_f_', 'high_f_', 'low_f_', 'volume_f_', 'avg_f_']]

        record_df['_open_%id' % self.freq] = 0

        log.info("writing data to: %s" % file_path)
        write_file(file_path, record_df.to_csv(), append=False)

    def read_data(self, context):
        file_info_path = self.data_path + 'info.txt'  # e.g.'data/ts_knn/info'
        file_list = read_file(file_info_path)

        logger.info('[after read] file_list = ' + file_list, log_level=0)
        file_list = file_list.split('\n')
        if len(file_list) == 0:
            return None

        file_list = file_list[:-1]
        current_dt = date2str(context.current_dt)

        def _isvalid(file_name):  # file_name = data/ts_time_select/20180102.csv
            ds = file_name.split('/')[-1].split('.')[0]
            return ds < current_dt

        file_list = list(filter(_isvalid, file_list))
        if len(file_list) == 0:
            return None

        def read_csv(file_name):
            body = read_file(file_name)
            return pd.read_csv(StringIO(body), index_col=0)

        logger.info('[after filter] file_list = ' + str(file_list), log_level=0)

        df_list = [read_csv(file_name) for file_name in file_list]
        df = pd.concat(df_list)
        return df

    def get_recent_factor_return(self, context,
                                 factors=['open_f_', 'close_f_', 'high_f_', 'low_f_', 'volume_f_', 'avg_f_']):
        factor_return_dict = dict()
        df = self.read_data(context)
        if df is None or df.shape[0] == 0:
            logger.info('data df is None', log_level=2)
            return factor_return_dict
        logger.info('file df shape = ' + str(df.shape), log_level=1)

        label = '_open_%id' % self.freq

        # top_5 factor return
        def _summarize(df, factor='avg_f_', group_col='date', label=label):
            group_list = df[group_col].unique()
            val_dict = {}
            for group in group_list:
                dff = df[df[group_col] == group]
                dff = dff.sort_index(by=factor, ascending=False)
                val_dict[group] = dff[label][:5].mean()
            return val_dict

        for factor in factors:
            val_dict = _summarize(df, factor=factor, group_col='date')

            x = sorted(list(map(str2date, map(str, val_dict.keys()))))
            y = [v for k, v in sorted(val_dict.items(), key=lambda item: item[0])]  # sort key

            if len(y) < self.recent_factor_return_win_size:
                factor_return_dict[factor] = None
            else:
                factor_return_dict[factor] = np.mean(y[-self.recent_factor_return_win_size:])
        return factor_return_dict


class KnnMinuteFactor:
    def __init__(self, context, g):
        self.context = context
        self.freq = 1

        self.pattern_len = 60
        self.knn_perception_threshold = 1.0 / 3
        self.date_weighting_fn = np.sqrt
        self.stock_weighting_fn = np.sqrt

        self.data_path = 'data/ts_knn/${date}.csv'

        self.stock_list = select_stocks(context)

    def run(self):
        pattern_df = self.pattern_scan()
        pattern_df = filter_abnormal(pattern_df)

        candidate_df = self.get_candidate_df()
        res_df = self.pattern_match(candidate_df, pattern_df)

        candidate_df = candidate_df.merge(res_df, on=['code', 'date'])

        candidate_df = candidate_df.sort_index(by='open_f_', ascending=False)
        return candidate_df

    @timeit(name='pattern_match (min)')
    def pattern_match(self, candidate_df, pattern_df):
        res_dict = {
            "code": [],
            "date": [],
            "open_f_": []
        }

        def rbf(x, sigma=1):
            x = np.array(x)
            return np.exp(- x ** 2 / (2 * sigma ** 2))

        def get_val(row, field, df):
            cols = Stream(self.pattern_len).map(lambda i: '%s_%im_' % (field, (-i))).tolist()
            my_x = row[cols]

            dis = df[cols].apply(lambda x: x * x).sum(axis=1) - 2 * df[cols].dot(my_x) + np.sum(my_x * my_x)
            dis = np.sqrt(np.array(dis, dtype=float))
            pattern_sigma = np.sqrt(df[cols].var().sum())  # center sigma

            dis = rbf(dis, sigma=pattern_sigma * self.knn_perception_threshold)

            # val = (dis * df['weight']).sum() / df['weight'].sum()  # weighted sum
            # incr = (df['_open_7d'] * dis * df['weight']).sum() / (dis * df['weight']).sum()
            val = (dis * df['weight'] * df['_open_%id' % self.freq]).sum() / df['weight'].sum()
            return val

        for row in trange(candidate_df):
            row = row[1]
            res_dict['code'].append(row['code'])
            res_dict['date'].append(row['date'])

            df = pattern_df[pattern_df['date'] < row['date']]

            for field in ["open"]:
                dis = get_val(row, field, df)
                res_dict[field + '_f_'].append(dis)

        res_df = pd.DataFrame(res_dict)
        return res_df

    @timeit(name='get_candidate_df (min)')
    def get_candidate_df(self):
        sb2 = SampleBuilder(index_code='000001.XSHG')
        sb2.set_stock_list(self.stock_list)

        sb2.generate_keys(
            index_code='000001.XSHG',
            end_date=self.context.current_dt,
            count=1,
            sample_dt=1
        )

        for field in ["open"]:
            sb2.filter_null()
            sb2.join_minute_price(
                field=field,
                count=self.pattern_len,
                execute_time="9:35",
                output_format=None)
        return sb2.df

    @timeit(name='pattern_scan (min)')
    def pattern_scan(self):
        end_date = self.context.current_dt - datetime.timedelta(days=self.freq + 1)
        sb = SampleBuilder(index_code='000001.XSHG')
        sb.set_stock_list(self.stock_list)

        sb.generate_keys(
            index_code='000001.XSHG',
            end_date=end_date,
            count=200 * 3,
            sample_dt=1
        )

        sb.join_price(
            dt_list=[self.freq],
            output_format="_open_%id",
            field="open",
            skip_online=False,
            to_incr=True
        )

        sb.df = sb.df[sb.df['_open_%id' % self.freq] > 0.1]
        sb.keys = sb.df[['code', 'date']]
        sb.df = sb.keys

        dt_list = Stream(self.freq).map(lambda x: x + 1).tolist()
        sb.join_price(
            dt_list=[dt for dt in dt_list],
            output_format="_open_%id",
            field="open",
            skip_online=False,
            to_incr=True
        )

        if self.freq > 3:
            sb.df['rankp'] = sb.df[Stream(range(self.freq)).map(lambda x: '_open_%id' % (x + 1)).tolist()].apply(rankp,
                                                                                                                 axis=1)
            sb.df = sb.df[sb.df['rankp'] > 0.99]

        for field in ["open"]:
            sb.filter_null()
            sb.join_minute_price(field=field,
                                 count=self.pattern_len,
                                 execute_time="9:35",
                                 output_format=None)

        def unique(x, to_df=True):
            pair = np.unique(x, return_counts=True)
            k = pair[0]
            v = pair[1]
            if not to_df:
                return dict(zip(k, v))
            df_kv = pd.DataFrame.from_dict({'value': k, 'count': v})
            return df_kv[['value', 'count']].sort_index(by='count', ascending=False)

        date_num_dict = unique(sb.df['date'].values, to_df=False)
        code_num_dict = unique(sb.df['code'].values, to_df=False)

        sb.df['date_num'] = sb.df['date'].apply(lambda s: date_num_dict.get(s, 0))
        sb.df['code_num'] = sb.df['code'].apply(lambda s: code_num_dict.get(s, 0))

        sb.df['weight'] = 1 / self.date_weighting_fn(sb.df['date_num']) / self.stock_weighting_fn(sb.df['code_num'])
        return sb.df


def select_stocks(context):
    sb = SampleBuilder(index_code='000001.XSHG')
    initial_list = get_stock_list(context)
    sb.set_stock_list(initial_list)
    sb.generate_keys(
        index_code='000001.XSHG',
        end_date=context.current_dt,
        count=1,
        sample_dt=1
    )
    sb.join_fundamental()
    df = sb.df
    df = df[df['pe_ratio'] > 0]
    # df = df.sort_index(by='market_cap')[:int(0.2*len(df))]
    df = df.sort_index(by='market_cap')[:500]
    stock_list = df['code'].unique()
    sb.logger.info("stock_list = %i / %i" % (len(stock_list), len(initial_list)))
    return stock_list


def get_stock_list(context):
    def _filter_name(name):
        not_st = 'ST' not in name
        not_star = '*' not in name
        return not_st and not_star

    def _filter_code(code):
        kc = code.startswith('688')
        cy = code.startswith('300')
        zx = code.startswith('002')
        return (not kc) and (not cy) and (not zx)

    stock_df = get_all_securities(types=['stock'])
    stock_df['code'] = stock_df.index
    stock_df = stock_df[stock_df['code'].apply(_filter_code)]
    stock_df = stock_df[stock_df['display_name'].apply(_filter_name)]

    stock_df['end_dt'] = stock_df['end_date'].apply(date2str)
    stock_df['start_dt'] = stock_df['start_date'].apply(date2str)

    end_dt = date2str(context.current_dt)
    start_dt = date2str(context.current_dt - datetime.timedelta(days=365 * 3))  # IPO >= 3 years

    stock_df = stock_df[stock_df['end_dt'] >= end_dt]
    stock_df = stock_df[stock_df['start_dt'] <= start_dt]

    initial_list = stock_df.index.tolist()

    panel = get_price(
        security=initial_list,
        start_date=context.current_dt,
        end_date=context.current_dt,
        frequency='daily',
        fields=['open', 'close', 'high', 'low', 'volume', 'money', 'avg',
                'high_limit', 'low_limit', 'paused'],
        skip_paused=False,
        fq='pre')

    price_df = panel[:, 0, :]

    stock_list = [
        stock for stock in initial_list if not (
                (price_df.loc[stock].open == price_df.loc[stock].high_limit) or
                (price_df.loc[stock].open == price_df.loc[stock].low_limit) or
                (price_df.loc[stock].paused != 0)
        )]

    return stock_list


