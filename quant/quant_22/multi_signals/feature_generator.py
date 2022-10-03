import numpy as np
import datetime
import sklearn
from sklearn.svm import SVR
from util import Stream, Logger, str2date
import copy

cross_sectional_features_config = {
    "val_f_": {
        "features": [
            "log_total_assets_",
            "log_total_liability_",
            "inc_revenue_year_on_year_",
            "log_development_expenditure_",
        ],
        "feature_group": [
            "industry_group"
        ],
        "label": "log_market_cap_"
    }
}


def init_cross_sectional_features_config():
    return copy.deepcopy(cross_sectional_features_config)


class FeatureGenerator:
    def __init__(self, df=None):
        self.df = df
        self.logger = Logger()

        self.feature_dt_list = None
        self.label_dt_list = None

        self.mode = None

        # attributes set in training
        self.norm_dict = {}

        self.feature_group = {
            "industry_group": [],
            "fundamental_group": ['market_cap', 'total_assets', 'total_liability', 'inc_revenue_year_on_year',
                                  'development_expenditure']
        }
        self.one_hot_field = {
            "industry_group": []
        }

    def set_df(self, df):
        self.df = df

    def set_logger(self, logger):
        self.logger = logger

    def log_level(self, log_level):
        self.logger.set_log_level(log_level)

    def set_feature_dt_list(self, feature_dt_list):
        self.feature_dt_list = feature_dt_list

    def set_label_dt_list(self, label_dt_list):
        self.label_dt_list = label_dt_list

    # ============= Main Functions ============= #
    def generate_features(self, mode='train'):
        self.logger.info('feature generation mode: %s' % mode)
        self.mode = mode
        self.generate_past_features()
        if mode != 'predict':
            self.generate_future_features()
        return self.df

    def generate_past_features(self):
        # TODO: modulize

        feature_dt_list = self.feature_dt_list
        df = self.df

        # ============= past price incr ============= #
        diff = Stream(feature_dt_list).filter(lambda dt: dt != 0) \
            .map(lambda dt: ('price_incr_d%i_' % (-dt), ["price_%id" % dt, "price_%id" % (dt + 1)])) \
            .to_dict()

        for k, v in diff.items():
            df[k] = (df[v[1]] - df[v[0]]) / df[v[0]]

        # ============= past index incr =========== #
        diff = Stream(feature_dt_list).filter(lambda dt: dt != 0) \
            .map(lambda dt: ('index_incr_d%i_' % (-dt), ["index_%id" % dt, "index_%id" % (dt + 1)])) \
            .to_dict()

        for k, v in diff.items():
            df[k] = (df[v[1]] - df[v[0]]) / df[v[0]]

        # =========== pass features ====== #
        cols = ["turnover_ratio",
                "pe_ratio",
                "pe_ratio_lyr",
                "pb_ratio",
                "ps_ratio",
                "pcf_ratio",

                'basic_eps',
                'diluted_eps',

                "roe",
                "roa",
                "net_profit_margin",
                "gross_profit_margin",
                "expense_to_total_revenue",
                "operation_profit_to_total_revenue",
                "ga_expense_to_total_revenue",
                "operating_profit_to_profit",
                "invesment_profit_to_profit",
                "inc_revenue_year_on_year",
                "inc_revenue_annual",
                "inc_operation_profit_year_on_year",
                "inc_operation_profit_annual",
                "inc_net_profit_year_on_year",
                "inc_net_profit_annual",

                #                 "minute_mean",
                #                 "minute_std"
                ]

        cols_ = Stream(cols).map(lambda s: s + '_').tolist()
        df[cols_] = df[cols].fillna(0)
        self.normalize(cols_)

        # ============= log features (volume & fundamental) ============= #
        cols = ['market_cap',
                'circulating_market_cap',

                'net_finance_cash_flow',
                'cash_from_invest',
                'goods_sale_and_service_render_cash',

                'total_assets',
                'total_liability',
                'development_expenditure',

                'operating_revenue',
                'total_operating_cost',
                'operating_cost',
                'financial_expense',
                'asset_impairment_loss',
                'investment_income',
                'operating_profit'
                ]

        cols += Stream(feature_dt_list).filter(lambda dt: dt != 0).map(lambda i: 'volume_%id' % i).tolist()
        log_cols = Stream(cols).map(lambda s: 'log_%s_' % s).tolist()

        self.log_cols = log_cols
        self.cols = cols

        df[log_cols] = df[cols].fillna(0).apply(lambda x: np.sign(x) * np.log1p(np.abs(x)))
        self.normalize(log_cols)

        # ============= one-hot features ============= #
        # 1. industry
        if self.mode == 'train':
            industry_code_list = df['industry_code'].unique().tolist()
            self.one_hot_field['industry_group'] = industry_code_list
            self.feature_group['industry_group'] = \
                Stream(industry_code_list).map(lambda industry_code: 'industry_%s_' % industry_code).tolist()
        else:
            industry_code_list = self.one_hot_field['industry_group']

        for industry_code in industry_code_list:
            df['industry_%s_' % industry_code] = (df['industry_code'] == industry_code).astype('int')

        # ============= special features ============= #
        self.generate_time_features()

        config = init_cross_sectional_features_config()
        # remark: if cross_sectional_features_config is used directly, pointers of list and sub-list will be passed
        #   and their elements will be modified, which will cause errors in multi-day operations like backtesting.
        self.generate_cross_sectional_features(config)

    def generate_future_features(self):
        label_dt_list = self.label_dt_list
        df = self.df

        # ============= future price incr ============= #
        diff = Stream(label_dt_list).filter(lambda dt: dt != 0) \
            .map(lambda dt: ('_price_incr_%id' % (dt), ["price_0d", "_price_%id" % dt])) \
            .to_dict()

        for k, v in diff.items():
            df[k] = (df[v[1]] - df[v[0]]) / df[v[0]]

        # ============= past index incr =========== #
        diff = Stream(label_dt_list).filter(lambda dt: dt != 0) \
            .map(lambda dt: ('_index_incr_%id' % (dt), ["index_0d", "_index_%id" % dt])) \
            .to_dict()

        for k, v in diff.items():
            df[k] = (df[v[1]] - df[v[0]]) / df[v[0]]

    # specific features
    def generate_cross_sectional_features(self, config):
        df = self.df
        for fea_name, params in config.items():
            fea_list = params['features']
            self.logger.info("fea_list = %s" % fea_list)
            for feature_group_name in params['feature_group']:
                fea_list += self.feature_group[feature_group_name]

            label = params['label']
            res = []

            for date in np.sort(df['date'].unique()):
                # self.logger.info('fea_name = %s, date = %s' % (fea_name, date))
                dff = df[df['date'] == date]
                # self.logger.info('fea_list = %s' % fea_list)
                X = dff[fea_list]
                Y = dff[label]
                model = SVR(kernel='rbf')
                model.fit(X, Y)
                pred = model.predict(X)
                err = pred - Y
                res += err.values.tolist()
            df[fea_name] = res

    def generate_time_features(self):
        df = self.df
        df['day_of_week'] = df['date'].apply(str2date).apply(datetime.datetime.weekday)
        df['day_of_year'] = df['date'].apply(lambda s: str2date(s).timetuple().tm_yday)

        df['week_sin_'] = np.sin(df['day_of_week'] / 7.0 * np.pi * 2)
        df['week_cos_'] = np.cos(df['day_of_week'] / 7.0 * np.pi * 2)

        df['year_sin_'] = np.sin(df['day_of_year'] / 365.0 * np.pi * 2)
        df['year_cos_'] = np.cos(df['day_of_year'] / 365.0 * np.pi * 2)

    # help functions
    def normalize(self, fea_list):
        """
        Normalize features in fea_list.

        Mean & std are calculated if normalization has never been done.
        Otherwise, use saved mean & std.

        TODO: feature-wise hash instead of fea_list-wise hash
        """
        hash_key = hash(str(fea_list))
        df = self.df

        if hash_key in self.norm_dict:
            mean, std = self.norm_dict[hash_key]
        else:
            mean = df[fea_list].mean()
            std = df[fea_list].std() + 1e-6
            self.norm_dict[hash_key] = (mean, std)
        df[fea_list] = (df[fea_list] - mean) / std


