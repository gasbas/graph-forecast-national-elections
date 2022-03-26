from torch_geometric.data import Data
from torch.utils.data import Subset
from torch_geometric.utils import from_networkx
import torch 
from tqdm import tqdm
import networkx as nx 
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')

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
        
        #Scaling features as they are the input of a Neural Network
        to_scale = [i for i in node_features.columns if i not in  ['reg_id','dep_id',"Resultat_t1", 
                                                                   "node_id"]]
        scale = node_features[to_scale].std() 
        node_features[to_scale] = (node_features[to_scale] -node_features[to_scale].mean()) / scale

        self.node_features = node_features
        
        # region ids and department ids will be processed by an embedding since
        # they are categorical features
        reg_ids = dict(zip(node_features['node_id'], node_features['reg_id']))
        dep_ids = dict(zip(node_features['node_id'], node_features['dep_id']))

        self.G = nx.read_weighted_edgelist('data/edgelist_dist10000_norm')
        node_features_ = dict(zip(node_features['node_id'], node_features[to_scale].values))

        nx.set_node_attributes(self.G, node_features_, 'x')
        nx.set_node_attributes(self.G, reg_ids, 'reg_id')
        nx.set_node_attributes(self.G, dep_ids, 'dep_id')
        
        # IMPORTANT: this mapper maps node_id in the graph G to their position in the 
        # G.nodes method. By default, creating a Data object from networkx will use the ordering
        # of G.nodes method, hence we need to define this mapper to have acces to the good indices.
        self.mapper = dict(zip([int(i) for i in list(self.G.nodes)], list(range(self.G.number_of_nodes()))))

    def fit(self, X_df, y) : 
        # The fit method only stores the y values to make it available
        # during transform timle
        self.y = torch.from_numpy(y)

        pass
    
    def transform(self, X) :
        # We create a Data object from PyTorch Geometric. It contains all the required data
        # to be processed by a GNN.
        
        ids = X['node_id'].astype(int).values
        # IMPORTANT: when creating a Data object with the from_networkx method,
        # the order of the nodes is the same as returned by the G.nodes method
        # In our case, the node ids are not ordered and we need to use a mapper
        # to move from node_id to position in the Data object. 
        data = from_networkx(self.G)
        
        # The mask represent the indices that will be used during training. More specifically
        # our GNN returns a N dimension output (where N = total number of municipalities)
        # but we only have access to the mask indices y values.
        data.mask = torch.LongTensor([self.mapper[i] for i in [int(i) for i in ids]])
        
        data.edge_index = data.edge_index.long()
        data.x = data.x.float()
        data.y = self.y
        
        #Returning a list is a little turnaround to make it work with ramp-test command.
        return [data] 