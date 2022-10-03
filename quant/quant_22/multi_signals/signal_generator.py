import numpy as np
import pandas as pd

from util import *
import sklearn
from sklearn.svm import SVR
import xgboost as xgb
import time

param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:linear'}


def get_null_row(df):
    null_row = df.isnull().any(axis=1)
    return np.where(null_row)


def filter_null_row(df, verbose=False):
    null_row = df.isnull().any(axis=1)
    if verbose:
        print('filter %i rows containing nan: %s' % (sum(null_row), np.where(null_row)))
    return df[~null_row]


def corr(df, signal='pred', label='_price_incr_7d'):
    not_null_idx = ~(df[signal].isnull() | df[label].isnull())
    return np.corrcoef(df[signal][not_null_idx], df[label][not_null_idx])[0, 1]


def topk(df, k=10, signal='pred', label='_price_incr_7d'):
    not_null_idx = ~(df[signal].isnull() | df[label].isnull())
    df = df[not_null_idx].sort_index(by=signal, ascending=False)
    return [df.iloc[:k, :][label].mean(), df[label].mean()]


class SignalGenerator:
    def __init__(self, df):
        self.df = df
        self.context = {}

        self.training_set = None
        self.test_set = None

        self.label = None

        self.features = None  # all factors
        self.features_ = None  # predictive factors
        self._features = None  # risk factors

        self.test_ratio = None

        self.model = None
        self.logger = Logger()

        self.rng = None
        self.fea_info = []

    def log_level(self, log_level):
        self.logger.set_log_level(log_level)

    def set_logger(self, logger):
        self.logger = logger

    def set_label(self, label):
        self.label = label

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def set_features(self, fea_list=None, sampling_rate=None):
        if fea_list is None:
            fea_list = Stream(self.df.columns).filter(lambda s: s[-1] == '_').tolist()

        if sampling_rate is not None:
            fea_num = int(len(fea_list) * sampling_rate)
            if self.rng is None:
                fea_list = np.random.choice(fea_list, fea_num, replace=False).tolist()
            else:
                fea_list = self.rng.choice(fea_list, fea_num, replace=False).tolist()

            # save random features
            self.fea_info.append({
                "fea_list": fea_list
            })

        self.features = fea_list
        self._features = Stream(fea_list).filter(lambda s: s[0] == '_').tolist()  # future features
        self.features_ = Stream(fea_list).filter(lambda s: s[0] != '_').tolist()  # past features

    def get_features(self):
        return self.features

    def select_features(self, bucket_num=20, threshold=0.99, verbose=False):
        """
        select statistical significant features by rank test (order-pair/inversions number significant)
        Return
            list of significant feature names
        """

        reward_dict = group_rewards(df=self.training_set, signals=self.features, label=self.label,
                                    bucket_num=bucket_num, verbose=verbose)
        reward_df = pd.DataFrame.from_dict(reward_dict, orient='index')

        reward_df['p-value'] = reward_df.apply(rankp, axis=1)
        reward_df['ip-value'] = reward_df.apply(irankp, axis=1)

        sig_df = reward_df[(reward_df['p-value'] > 0.99) | (reward_df['ip-value'] > 0.99)][['p-value', 'ip-value']]
        self.features = sig_df.index.tolist()
        return self.features

    def split(self, test_ratio=None, split_point=None, split_field='date', verbose=False):
        """
        do training set - test set split on split_field
        Args:
            test_ratio: test set ratio, will be override if split_point is not None
            split_point: should be of the same type as df[split_field]
            split_field: must be in df.columns
        Returns:
            generate self.training_set, self.test_set
        """
        df = self.df
        sample_num = float(df.shape[0])

        if split_point is None:
            df = df.sort_index(by=split_field, ascending=True)
            split_point = df[split_field].iloc[int(sample_num * (1 - test_ratio))]

        self.training_set = df[df[split_field] <= split_point]
        self.test_set = df[df[split_field] > split_point]

        train_num = self.training_set.shape[0]
        test_num = self.test_set.shape[0]

        if verbose:
            self.logger.info("training set: sample num = %s, raito = %.3f" % (train_num, train_num / sample_num))
            self.logger.info("test set: sample num = %s, ratio = %.3f" % (test_num, test_num / sample_num))

    @timeit
    def fit_svr(self,
                kernel='rbf',
                verbose=True
                ):

        features, label = self.features, self.label

        training_set = filter_null_row(self.training_set[features + [label]])

        assert training_set is not None

        X = np.array(training_set[features].values)
        Y = np.array(training_set[label].values)

        model = sklearn.svm.SVR(kernel=kernel, verbose=verbose)
        model.fit(X, Y)
        self.model = model

    def predict_svr(self, df, mode='predict'):
        assert self.model is not None
        assert set(self.features_) <= set(self.df.columns.tolist())

        if mode == 'predict':
            tmp_df = df.copy()
            tmp_df[self._features] = 0

            null_row = get_null_row(tmp_df)
            tmp_df = tmp_df.fillna(0)
            X = tmp_df[self.features]

            return self.model.predict(X)

        if mode == 'train':
            raise NotImplementedError()

    def fit_xgboost(self,
                    num_round=2,
                    params=None):

        self.logger.info('Start fitting ...')
        start_time = time.time()

        features, label = self.features, self.label

        training_set = filter_null_row(self.training_set[features + [label]])

        assert training_set is not None

        dtrain = xgb.DMatrix(training_set[features], label=training_set[label])
        if params is None:
            params = {'max_depth': 5, 'eta': 1, 'objective': 'reg:linear', 'silent': 0}

        model = xgb.train(param, dtrain, num_round)
        self.model = model

        self.logger.info('Finish fitting! fit time: %.2f' % (time.time() - start_time))

    def predict_xgboost(self, df, mode='predict'):
        assert self.model is not None
        assert set(self.features_) <= set(self.df.columns.tolist())

        if mode == 'predict':
            tmp_df = df.copy()
            tmp_df[self._features] = 0

            null_row = get_null_row(tmp_df)
            tmp_df = tmp_df.fillna(0)
            X = tmp_df[self.features]
            X = xgb.DMatrix(X)

            return self.model.predict(X)


