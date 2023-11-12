import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GINConv, PNAConv, SAGEConv, ChebConv, MLP
import torch
import sys
from build_graph import ordered_word_pair

# GRAPH Models

class GCN(nn.Module):
    def __init__(self, gnn_layers , in_channels, hidden_channels, out_channels):
        super().__init__()
        self.module = [GCNConv(in_channels, hidden_channels) ]

        self.module = self.module + [ GCNConv(hidden_channels, hidden_channels) for i_ in range(gnn_layers - 2) ]

        self.module.append( GCNConv(hidden_channels, out_channels)  )

        self.module = nn.ModuleList(self.module )

    def forward(self, x, edge_index, edge_weight) :
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
 
        for i in range(len(self.module) - 1) :
            x = self.module[i](x, edge_index, edge_weight).relu()
        x = self.module[-1](x, edge_index, edge_weight)
        return x


class GAT(nn.Module):
    def __init__(self, gnn_layers , in_channels, hidden_channels, out_channels):
        super().__init__()
        self.module = [GATConv(in_channels, hidden_channels) ]

        self.module = self.module + [ GATConv(hidden_channels, hidden_channels) for i_ in range(gnn_layers - 2) ]

        self.module.append( GATConv(hidden_channels, out_channels)  )

        self.module = nn.ModuleList(self.module )

    def forward(self, x, edge_index, edge_weight) :
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]

        for i in range(len(self.module) - 1) :
            x = self.module[i](x, edge_index,edge_weight).relu() # edge_weight is considred as edge attributes
        x = self.module[-1](x, edge_index)
        return x


class GATV2(nn.Module):
    def __init__(self, gnn_layers , in_channels, hidden_channels, out_channels):
        super().__init__()
        self.module = [GATv2Conv(in_channels, hidden_channels,edge_dim=1) ]

        self.module = self.module + [ GATv2Conv(hidden_channels, hidden_channels,edge_dim=1) for i_ in range(gnn_layers - 2) ]

        self.module.append( GATv2Conv(hidden_channels, out_channels,edge_dim=1)  ) 

        self.module = nn.ModuleList(self.module )

    def forward(self, x, edge_index, edge_weight) :
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]

        for i in range(len(self.module) - 1) :
            x = self.module[i](x=x, edge_index=edge_index, edge_attr = edge_weight.unsqueeze(1)).relu() # edge_weight is considred as edge attributes
        x = self.module[-1](x, edge_index)
        return x


class GIN(nn.Module):
    def __init__(self, gnn_layers , in_channels, hidden_channels, out_channels):
        super().__init__()
        self.module = [GINConv(nn=MLP([in_channels, hidden_channels, hidden_channels]), train_eps=False) ]

        self.module = self.module + [ GINConv(nn=MLP([hidden_channels, hidden_channels, hidden_channels]), train_eps=False) for i_ in range(gnn_layers - 2) ]

        self.module.append( GINConv( nn=MLP([hidden_channels, hidden_channels, out_channels]), train_eps=False))  

        self.module = nn.ModuleList(self.module )

    def forward(self, x, edge_index, edge_weight) :
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        # x = x.to_dense()
        for i in range(len(self.module) - 1) :
            
            x = self.module[i](x, edge_index.to_dense()).relu()
        x = self.module[-1](x, edge_index)
        return x

   
class PNA(nn.Module):
    def __init__(self, gnn_layers , in_channels, hidden_channels, out_channels):
        super().__init__()
        self.module = [PNAConv(in_channels, hidden_channels) ]

        self.module = self.module + [ PNAConv(hidden_channels, hidden_channels) for i_ in range(gnn_layers - 2) ]

        self.module.append( PNAConv(hidden_channels, out_channels)  )

        self.module = nn.ModuleList(self.module )

    def forward(self, x, edge_index, edge_weight) :
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]

        for i in range(len(self.module) - 1) :
            x = self.module[i](x, edge_index, edge_weight).relu()
        x = self.module[-1](x, edge_index, edge_weight)
        return x

        
class GraphSAGE(nn.Module):
    def __init__(self, gnn_layers , in_channels, hidden_channels, out_channels):
        super().__init__()
        self.module = [SAGEConv(in_channels, hidden_channels) ]

        self.module = self.module + [ SAGEConv(hidden_channels, hidden_channels) for i_ in range(gnn_layers - 2) ]

        self.module.append( SAGEConv(hidden_channels, out_channels)  )

        self.module = nn.ModuleList(self.module )

    def forward(self, x, edge_index, edge_weight) :
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = x.to_dense()
        for i in range(len(self.module) - 1) :
            x = self.module[i](x, edge_index).relu()
        x = self.module[-1](x, edge_index)
        return x


class Cheb(nn.Module):
    def __init__(self, gnn_layers , in_channels, hidden_channels, out_channels):
        super().__init__()
        self.module = [ChebConv(in_channels, hidden_channels,K=1) ]

        self.module = self.module + [ ChebConv(hidden_channels, hidden_channels,K=1) for i_ in range(gnn_layers - 2) ]

        self.module.append( ChebConv(hidden_channels, out_channels,K=1)  )

        self.module = nn.ModuleList(self.module )

    def forward(self, x, edge_index, edge_weight) :
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        for i in range(len(self.module) - 1) :
            x = self.module[i](x, edge_index, edge_weight).relu()
            
        x = self.module[-1](x, edge_index, edge_weight)
        return x

               
# SEQUENTIAL Models
class LSTM_w_activations(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers) :

        super(LSTM_w_activations, self).__init__()
         
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = self.input_size ,hidden_size = self.hidden_size , num_layers = self.num_layers, batch_first = True , bidirectional = True)

    def forward(self, x):
        output = self.lstm(x)
        return output


# Debug agg
class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(512,128)
        self.fc2 = nn.Linear(128,1)
        self.relu = torch.nn.ReLU() # instead of Heaviside step fn
    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output) # instead of Heaviside step fn
        output = self.fc2(output)
        output = torch.sigmoid(output)
        return output


# Aggregation
class MLP_agg(nn.Module):
    def __init__(self):
        super(MLP_agg, self).__init__()
        self.fc1 = nn.Linear(1024,128)
        self.fc2 = nn.Linear(128,1)
        self.relu = nn.ReLU() # instead of Heaviside step fn
    def forward(self, x):
        x = x.squeeze(0)
        x = torch.cat([x[0,:], x[-1,:]], dim = -1)

        output = self.fc1(x)
        output = self.relu(output) # instead of Heaviside step fn
        output = self.fc2(output)
        output = torch.sigmoid(output)
        return output




# Aggregation
class Sum_agg(nn.Module):
    def __init__(self):
        super(Sum_agg, self).__init__()
        self.fc1 = nn.Linear(512,128)
        self.fc2 = nn.Linear(128,1)
        self.relu = nn.ReLU() # instead of Heaviside step fn
    def forward(self, x):
        x = x.squeeze(0)
        x = torch.sum(x, dim=0)
        output = self.fc1(x)
        output = self.relu(output) # instead of Heaviside step fn
        output = self.fc2(output)
        output = torch.sigmoid(output)
        return output
    

# Aggregation
class AVG_agg(nn.Module):
    def __init__(self):
        super(AVG_agg, self).__init__()
        self.fc1 = nn.Linear(512,128)
        self.fc2 = nn.Linear(128,1)
        self.relu = nn.ReLU() # instead of Heaviside step fn
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
    def forward(self, x):
        x = x.squeeze(0)
        x = torch.mean(x, dim=0)
        output = self.fc1(x)
        output = self.relu(output) # instead of Heaviside step fn
        output = self.fc2(output)
        output = torch.sigmoid(output)
        return output



class Prod_agg(nn.Module):
    def __init__(self, window_size):
        super(Prod_agg, self ).__init__()
        self.window_size = window_size
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,128)
        self.relu = nn.ReLU() # instead of Heaviside step fn
    def forward(self, x ):

        x = x.squeeze(0)
        output = self.fc1(x)
        output = self.relu(output) # instead of Heaviside step fn
        output = self.fc2(output)
        scores = torch.sigmoid(output @ output.T)
        scores_mask = torch.triu(torch.triu(torch.ones(output.size(0) , output.size(0)).to(output.device), diagonal=1).T , diagonal=-self.window_size ).T
        predicted_prob =torch.prod(scores * scores_mask + (1-scores_mask) )

        return predicted_prob.unsqueeze(0)

