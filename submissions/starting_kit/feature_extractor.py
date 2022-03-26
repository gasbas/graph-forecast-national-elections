import pandas as pd
import numpy as np 
import os 

class FeatureExtractor(object) : 
    def __init__(self)  :
        region_df = pd.read_csv('data/region.csv')
        node_features = pd.read_csv('data/node_features.csv', low_memory = False)
        
        node_features['node_id'] = node_features['node_id'].astype(str)
        node_features = node_features.merge(region_df, on = 'reg_id')
        node_features['Y_t1_macron'] = node_features['Resultat_t1'].apply(lambda x: eval(x)['Emmanuel MACRON'])
        node_features['Y_t1_lepen'] = node_features['Resultat_t1'].apply(lambda x: eval(x)['Marine LE PEN'])
        voters = node_features['Y_t1_macron'] + node_features['Y_t1_lepen']

        node_features['Y_t1_macron'] /= voters
        node_features['Y_t1_lepen'] /= voters

        node_features[['Y_t1_macron','Y_t1_lepen']] = node_features[['Y_t1_macron','Y_t1_lepen']].fillna(0)
        node_features.drop(['Resultat_t1'], axis = 1, inplace=True)
        self.node_features = node_features
    
    def fit(self, X_df, y) : 

        pass

    def transform(self, X) :
        X = self.node_features.loc[X.index]
        
        return X.drop(['node_id'],axis=1)