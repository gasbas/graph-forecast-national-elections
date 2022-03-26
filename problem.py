import os
import string
from glob import glob

import pandas as pd
import numpy as np

import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import BaseCrossValidator

class RegionalSplit(BaseCrossValidator):

    def __init__(self, n_splits = 10, reg_col='reg_id', test_size = 0.5, random_state=None):  
        self.reg_col = reg_col
        self.test_size = test_size
        self.n_splits = n_splits
        if random_state : 
            self.gen = np.random.default_rng(random_state)
        else : 
            self.gen = np.random.default_rng(np.random.randint(1e10))

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits

    def split(self, X, y = None, groups=None):

        n_splits = self.get_n_splits(X, y, groups)
        for i in range(n_splits):
            assert isinstance(X, pd.DataFrame), "Provided data must be a dataframe"
            assert self.reg_col in X.columns
            indices = X.groupby(self.reg_col).indices
            train_nodes = []
            test_nodes = []
            for k,v in indices.items(): 
                test_size_abs = int(len(v) * self.test_size) 
                random_choice = self.gen.choice(len(v), len(v), replace=False)
                test_indices = v[random_choice[:test_size_abs]]
                train_indices =  v[random_choice[test_size_abs:]]
                
                train_nodes.extend(train_indices.tolist())
                test_nodes.extend(test_indices.tolist())
            yield (
                np.array(train_nodes), np.array(test_nodes)
            )

problem_title = "Forecasting secound round of national elections based on Graphs"

label_name = ['y']

Predictions = rw.prediction_types.make_regression(label_names=label_name)
workflow = rw.workflows.FeatureExtractorRegressor()


class MAE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="MAE", precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred, weights = None):
        mae =  np.average((np.abs(y_true - y_pred)), weights = weights)
        return mae


score_types = [
    MAE(name="MAE"),
]

def _get_data(path=".", split="train"):

    nodes = []
    if split == 'train': 
        node_file = 'train_nodes'
    else : 
        node_file = 'test_nodes'
    
    with open(os.path.join(path,f'data/{node_file}.txt'),'r') as f :
        for line in f.readlines(): 
            nodes.append(line.split('\n')[0])

    node_features = pd.read_csv(os.path.join(path, 'data/node_features.csv'))
    node_features['node_id'] = node_features['node_id'].astype(str)

    y = pd.read_csv('data/y.csv', index_col = 0)
    y['node_id'] = y['node_id'].astype(str)
    y = y[y['node_id'].isin(nodes)]['y'].values

    X = node_features[node_features['node_id'].isin(nodes)][['node_id', 'reg_id']]
    
    return X, y.reshape(-1,1)


def get_train_data(path="."):
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")


def get_cv(X, y):
    cv = RegionalSplit(n_splits=10, test_size=0.5, random_state=2022, reg_col = 'reg_id')
    return cv.split(X, y)
