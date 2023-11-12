import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
device = f'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
dataset = PygNodePropPredDataset(name='ogbn-products',
                                 transform=T.ToSparseTensor())
data = dataset[0]
# this dataset comes with train-val-test splits predefined for benchmarking
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

print(f' dataset has {data.num_nodes} nodes where each node has a {data.num_node_features} dim feature vector')
print(f' dataset has {data.num_edges} edges where each edge has a {data.num_edge_features} dim feature vector')
print(f' dataset has {dataset.num_classes} classes')

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)
    
    def forward(self, data):
        x = self.conv1(data.x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv2(x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv3(x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        return torch.log_softmax(x, dim=-1)


print(type(data.x))
print(type(data.adj_t))

print(data.x.shape)
print(data.adj_t)
#lr = 1e-4 
#epochs = 50 
#hidden_dim = 75
#model = GraphSAGE(in_dim=data.num_node_features, hidden_dim=hidden_dim, out_dim=dataset.num_classes)
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# compute activations for train subset
#out = model(data)[train_idx]
#print(out)