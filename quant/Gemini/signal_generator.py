import numpy as np
import pandas as pd

from util import Stream
import sklearn
from sklearn.svm import SVR
import xgboost as xgb


def get_null_row(df):
    null_row = df.isnull().any(axis=1)
    return np.where(null_row)


def filter_null_row(df):
    null_row = df.isnull().any(axis=1)
    print('filter %i rows containing nan: %s' % (sum(null_row), np.where(null_row)))
    return df[~null_row]


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

    def set_label(self, label):
        self.label = label

    def set_features(self, fea_list=None):
        if fea_list is None:
            fea_list = Stream(self.df.columns).filter(lambda s: s[-1] == '_').tolist()

        self.features = fea_list
        self._features = Stream(fea_list).filter(lambda s: s[0] == '_').tolist()  # future features
        self.features_ = Stream(fea_list).filter(lambda s: s[0] != '_').tolist()  # past features

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
            print("training set: sample num = %s, raito = %.3f" % (train_num, train_num / sample_num))
            print("test set: sample num = %s, ratio = %.3f" % (test_num, test_num / sample_num))

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

    def fit_xgboost(self,
                    num_round=2,
                    params=None):
        features, label = self.features, self.label

        training_set = filter_null_row(self.training_set[features + [label]])

        assert training_set is not None

        dtrain = xgb.DMatrix(training_set[features], label=training_set[label])
        if params is None:
            params = {'max_depth': 5, 'eta': 1, 'objective': 'reg:linear', 'silent': 0}

        model = xgb.train(param, dtrain, num_round)
        self.model = model

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

    @staticmethod
    def select_feature(df):
        fea_list = df.columns.tolist()

        summary_df = df.describe()
        summary_df['valid'] = (summary_df['max'] > summary_df['min'])  # filter all same column

        df = df[summary_df['valid']]  # select valid rows
        self.features = Streams(fea_list).filter(lambda s: s in df.index.tolist()).tolist()

        raise NotImplementedError

    def predict(self, df, mode='predict'):
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

    def evaluate(self):
        pass


sg = SignalGenerator(df_train)
sg.split(0.2, verbose=True)

sg.set_label('_price_incr_7d')
sg.set_features()

start_time = time.time()
# sg.fit_svr()
params = {'max_depth': 5, 'eta': 1, 'objective': 'reg:linear', 'silent': 0}
sg.fit_xgboost(num_round=100, params=params)
print("\nfit svr time elaspsed: %.2f" % (time.time() - start_time))

# df_predict['pred'] = sg.predict(df_predict)
# sg.test_set['pred'] = sg.predict(sg.test_set)

df_predict['pred'] = sg.predict_xgboost(df_predict)
sg.test_set['pred'] = sg.predict_xgboost(sg.test_set)