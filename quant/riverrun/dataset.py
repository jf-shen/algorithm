import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn 
import sklearn


class DataSet(pd.DataFrame):
    context = {} 
    display_columns = ['stock_name', 'date', 'val_f_', 'pred_', '_pred'] 
    
    training_set = None
    test_set = None 
    
    model = 'linear_model'
    label = None
    test_ratio = 0.2
    
    model_obj = None
    
    @property
    def _constructor(self):
        return DataSet
    
    def __iadd__(self, df):
        self = self.append(df)
        return self
    
    
    # ================ Quick Functions =============== # 
    
    def head_(self, n=3):
        return self[self.display_columns].head(n)
    
    def tail_(self, n=3):
        return self[self.display_columns].tail(n)
    
    def mse(self, pred=None, label=None):
        if label is None:
            assert self.label is not None, "no label and no default label"
            label = self.label
            print("use default label: ", label)
        
        return np.mean((self[pred] - self[label]) ** 2)
    
    
    def summary(self):
        display_columns = list(filter(lambda s: s in self.display_columns, self.columns))
        df = self[display_columns]
        summary_df = df.describe()
        return summary_df
    
    def get_label(self):
        return self[self.label]
    
    def add_display(self, *args):
        for s in args:
            self.display_columns.append(s)
            
            
    # =============== Ancillary Functions ============= # 
    
    def check_label(self, label, verbose=False):
        assert (label is not None) or (self.label is not None), 'no label and default label'
        if label is not None:
            self.label = label 
        else:
            label = self.label
            if verbose:
                print("use default label: ", label)
        return label
        
    def check_model(self, model, verbose=False):
        assert (model is not None) or (self.model is not None), 'no model and default model'
        if model is not None:
            self.model = model
        else:
            model = self.model
            if verbose:
                print("use default model: ", model)
        return model
    
    def check_test_ratio(self, test_ratio, verbose=False):
        assert (test_ratio is not None) or (self.test_ratio is not None), \
                'no test ratio and default test ratio '
        if test_ratio is not None:
            self.test_ratio = test_ratio
        else:
            test_ratio = self.test_ratio
            if verbose:
                print("use default test_ratio: ", test_ratio)
        return test_ratio
        
    
    # ================ Predictive Models  =============== # 
    
    def split(self, test_ratio=None, split_point=None, split_field='date'):
        """
        do training set - test set splitting by split_point of split_field
        Args:
            test_ratio: test set ratio, will be override if split_point is not None
            split_point: should be of the same type as df[split_field]
            split_field: must be in df.columns
        Returns:
            self.training_set, self.test_set 
        """
        
        assert split_field in self.columns, "field is not found: %s" % split_field
        
        test_ratio = self.check_test_ratio(test_ratio)
        
        df = self
        sample_num = df.shape[0] 
        if split_point is None:
            df = df.sort_values(by=split_field, ascending=True)
            split_point = df[split_field][int(sample_num * (1 - test_ratio))]
        
        self.training_set = df[df[split_field] <= split_point]
        self.test_set = df[df[split_field] > split_point]
        
        train_num = self.training_set.shape[0] 
        test_num = self.test_set.shape[0]
        
        sample_num = float(sample_num)
        print("training set: sample num = %s, raito = %.3f" % (train_num, train_num / sample_num))
        print("test set: sample num = %s, ratio = %.3f" % (test_num, test_num / sample_num)) 
  
    
    def fit(self, 
            features=[], 
            label=None, 
            model = None, 
            *args, 
            **kargs) :

        label = self.check_label(label, verbose=True)
        model = self.check_model(model)
        print('model: %s' % model)
        
        if model == 'linear_model':
            return self.linear_model(features=features, label=label, *args, **kargs)
        elif model == 'elastic_net':
            return self.elastic_net(features=features, label=label, *args, **kargs)
        elif model == 'svr':
            return self.svr(features=features, label=label, *args, **kargs)
        else:
            raise NotImplementedError(model)  
    
    def svr(self, 
            features=[], 
            label=None, 
            kernel='rbf', 
            verbose=False, 
            suffix='',
            append_prediction=True
        ):
        
        training_set, test_set = self.training_set, self.test_set
        assert (training_set is not None) and (test_set is not None)
        
        X = np.array(training_set[features].values)
        Y = np.array(training_set[label].values) 
        
        model = sklearn.svm.SVR(kernel=kernel, verbose=verbose)
        model.fit(X, Y)
        self.model_obj = model
        
        def _mask(X):
            # mask unknown features
            idx = [fea[0] == '_' for fea in features]
            X_mask = X.copy() 
            X_mask[:, idx] = 0 
            return X_mask 
        
        if append_prediction:
            self['_pred%s' % suffix] = model.predict(self[features]) 
            self['pred%s_' % suffix] = model.predict(_mask(self[features].values))
            self['_err%s' % suffix] = np.array(self[label] - self['_pred%s' % suffix])
            self.split()
            
    def elastic_net(self, 
            features=[], 
            label=None, 
            alpha=1e-4,
            l1_ratio=0.5,
            suffix='',
            append_prediction=True,
            fit_intercept=False
        ):
        
        training_set, test_set = self.training_set, self.test_set
        assert (training_set is not None) and (test_set is not None)
        
        X = np.array(training_set[features].values)
        Y = np.array(training_set[label].values)
        
        model = sklearn.linear_model.ElasticNet(alpha=alpha, 
                                                l1_ratio=l1_ratio, 
                                                fit_intercept=fit_intercept)
        model.fit(X, Y)
        self.model_obj = model
        
        
        coeff_dict = dict(zip(features, model.coef_))
        if fit_intercept:
            coeff_dict['intercept'] = model.intercept_
            
        def _mask(X):
            # mask unknown features
            idx = [fea[0] == '_' for fea in features]
            X_mask = X.copy() 
            X_mask[:, idx] = 0
            return X_mask 
            
        if append_prediction:
            self['_pred%s' % suffix] = model.predict(self[features]) 
            self['pred%s_' % suffix] = model.predict(_mask(self[features].values))
            self['_err%s' % suffix] = np.array(self[label] - self['_pred%s' % suffix])
            self.split()
            
        coeff_df = pd.DataFrame.from_dict(coeff_dict, orient='index')
        coeff_df.columns = ['coeff']
        coeff_df = coeff_df.reindex(index = features)
        return coeff_df
        
        
    def linear_model(self, 
                     features=[], 
                     label=None, 
                     l2=1e-4, 
                     append_prediction=True, 
                     suffix=''):
        
        training_set, test_set = self.training_set, self.test_set
        assert (training_set is not None) and (test_set is not None)
            
        X = np.array(training_set[features].values)
        Y = np.array(training_set[label].values) 
        
        _, d = X.shape
        
        # explanatory model: with future features
        beta = np.linalg.inv(X.T.dot(X) + l2 * np.eye(d)).dot(X.T).dot(Y)
        coeff_dict = dict(zip(features, beta))
        
        # predictive model: no future features
        features_ = list(filter(lambda s: s[0] != '_', features)) 
        beta_ = np.array([coeff_dict[fea] for fea in features_])
        

        def mse(df):
            pred = np.array(df[features].dot(beta))
            Y = np.array(df[label])
            return np.mean((Y - pred) ** 2) 
        
        def mse_(df):
            pred_ = np.array(df[features_].dot(beta_))
            Y = np.array(df[label])
            return np.mean((Y - pred_) ** 2) 
               
        
        print("training explanatory MSE: %.6f " % mse(training_set))
        print("test set explanatory MSE: %.6f" % mse(test_set))
        
        print("training predictive MSE: %.6f" % mse_(training_set))
        print("test set predictive MSE: %.6f" % mse_(test_set))

        # append prediction 
        if append_prediction:
            self['_pred%s' % suffix] = np.array(self[features]).dot(beta)
            self['pred%s_' % suffix] = np.array(self[features_]).dot(beta_)
            self['_err%s' % suffix] = np.array(self[label] - self['_pred%s' % suffix])
            self.split()

        coeff_df = pd.DataFrame.from_dict(coeff_dict, orient='index')
        coeff_df.columns = ['coeff']
        coeff_df = coeff_df.reindex(index = features)
        return coeff_df
    
    # ================ Analysis  =============== # 
    
    def get_reward(self, signal, a_min=None, a_max=None, label=None, mode='test'):
        label = self.label if label is None else label 
        assert label is not None
        self.label = label
        
        if mode == 'test':
            df = self.test_set
        elif mode == 'train':
            df = self.training_set 
        else:
            df = self
        
        a_min = -np.inf if a_min is None else a_min 
        a_max = np.inf if a_max is None else a_max
        
        df['is_candidate'] = (df[signal] >= a_min) & (df[signal] < a_max)
        return df[df['is_candidate']][label].mean()
    
    def plot_rewards(self, signal, percentage=True, label=None, mode='test', bucket_num=100, auto_split=True):
        if auto_split:
            self.split()
        if type(signal) == list:
            for s in signal:
                self.plot_rewards(s, percentage, label, mode, bucket_num, False)
            plt.title('top-k %s average reward (%s set)' % (signal, mode))
            plt.xlabel(str(signal) + ' percentage')
            plt.legend('best', labels=signal)
        else:
            signal_list =  np.sort(self[signal])
            x_list = []
            y_list = []
            percent_list = [] 
            
            for i in range(bucket_num):
                x_min = signal_list[int(i / float(bucket_num) * len(signal_list))]
                x_max = signal_list[int((i + 1) / float(bucket_num) * len(signal_list)) - 1]
                y = self.get_reward(signal, a_min=x_min, a_max=x_max, label=label, mode=mode)
                
                percent_list.append(i / float(bucket_num) * 100)
                x_list.append(x_min)
                y_list.append(y)
            
            plt.title('[%s] average reward (%s set)' % (signal, mode)) 
            if percentage:
                plt.scatter(percent_list, y_list, s = 10, label=signal)
                plt.xlabel(signal + ' percentage')
            else:
                plt.scatter(x_list, y_list, s = 10, label=signal)
                plt.xlabel(signal)
            plt.ylabel('reward')

    def hist(self, field, a_min=None, a_max=None, bins=20):
        """
        histogram a field in columns 
        Args:
            field: a column name 
            a_min: valuse < a_min are truncated
            a_max: values > a_max are truncated
            bins: number of bars 
        """
        x = self[field]
        a_min = min(x) if a_min is None else a_min
        a_max = max(x) if a_max is None else a_max
        plt.hist(self[field].clip(a_min, a_max).values, bins=bins)
        plt.xlabel(field)
        plt.title(field)
        
    def hist_err(self, bins=20):
        """
        histogram the '_err' field which is the prediction error for k-day price increase
        """
        self.hist('_err', a_min=-0.1, a_max=0.1, bins=bins)
        
    def discretize(self, fea_name, bucket_num, cum=False):
        """
        discretize a feature and the discretized features are added to columns
        Args:
            fea_name: the column name of the feature to be discretized
            bucket_num: chunks to be divided
        """
        x = np.sort(self[fea_name]) 
        for i in range(bucket_num):
            if fea_name[-1] == '_':
                name = fea_name[:-1]
                suffix = '_'
            else:
                name = fea_name
                suffix = ''
            
            col_name = name + '_%.1f%%' % (i / bucket_num * 100) + suffix
            x_min = x[int(i / bucket_num * len(x))]
            x_max = x[int((i + 1) / bucket_num * len(x) - 1)]
            
            col_val = np.zeros(len(x))
            if cum:
                col_val[self[fea_name] < x_max] = 1 
            else:
                col_val[(self[fea_name] >= x_min) & (self[fea_name] < x_max)] = 1 

            self[col_name] = col_val

            
   