from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
import pandas as pd 
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GCN(torch.nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.feat_map1 = torch.nn.Linear(input_features, 32)

        self.reg_emb = torch.nn.Embedding(22, 16)
        self.dep_emb = torch.nn.Embedding(96, 16)

        self.conv1 = SAGEConv(32 + 16*2 , 16, normalize = False)
        self.conv2 = SAGEConv(16, 32, normalize = False)
        self.lin = torch.nn.Linear(32, 1)


    def forward(self, data, train = True):
        x, edge_index = data.x, data.edge_index
        reg_ids = data.reg_id
        dep_ids = data.dep_id
        
        reg_fm = self.reg_emb(reg_ids)
        dep_fm = self.dep_emb(dep_ids)
        
        # First project input data by a linear feature map
        x = self.feat_map1(x)
        
        # Concatenation with departement and region embeddings to 
        # create our GNN input
        x = torch.cat([x, reg_fm, dep_fm],1)
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        
        return x
    
    
class Regressor(BaseEstimator):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, data, Y):
        data = data[0]
        self.model = GCN(data.num_node_features).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay = 5e-3)
        self.criterion = torch.nn.L1Loss()
        
        data = data.to(self.device)
        self.model.train()
        pbar = tqdm(range(500))
        for epoch in pbar:
            total_loss = 0 

            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out[data.mask], data.y.float())
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_description(f'Epoch {epoch+1}, Loss : {(total_loss):0.4f}')
        
    def predict(self, data):
        with torch.no_grad() : 
            data = data[0].to(self.device)
            self.model.eval()
            out = self.model(data)
            out = (out[data.mask])
            
            return out.detach().cpu().numpy()