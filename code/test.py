import torch
torch.manual_seed(0)

import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool, global_add_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from torch_geometric.utils import add_self_loops, degree, to_dense_adj
from torch_geometric.data import Data
from torch_geometric.nn import MLP, GINConv, global_add_pool

import json
import sys
import copy
import numpy as np
import pandas as pd
import argparse

from torch_geometric.utils import dropout_adj

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))    
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                        dropout=0.5)

    def forward(self, x, edge_index, batch):
        
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)


    def predict(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        h = x
        x = global_add_pool(x, batch)
        return self.mlp(x), h

def train(model, optimizer, train_loader, loss_function, device = torch.device("cuda")):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        print(out)
        #loss = F.nll_loss(out, data.y)
        loss = loss_function(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


def test(model, loader, device = torch.device("cuda")):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.batch).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name_dataset', type = str, default='PROTEINS', help='Dataset to use')
    parser.add_argument('--dropout_edge', type = float, default=0.5, help='Dropout probability of edge')
    args = parser.parse_args(args = [])

    name_data = args.name_dataset

    print('We are at data : {}' .format(name_data))

    dataset = TUDataset(root='/tmp/' + name_data, name=name_data)
    num_class = dataset.num_classes

    loss_function = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    input_dim = dataset[0].x.shape[1]

    input_dim = dataset[0].x.shape[1]

    dataset = dataset.shuffle()
    num_samples = len(dataset)
    batch_size = 32
    num_val = num_samples // 10
    val_dataset = dataset[:num_val]
    test_dataset = dataset[num_val:2 * num_val]
    train_dataset = dataset[2 * num_val:]
    num_train_epochs = 101

    model = Net(dataset.num_features, 64, dataset.num_classes, 3).to(device)

    batch_size = 32
    num_folds = 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #model = GCN(input_dim, [32, 32],32 , 2, 0.2).to(device)
    model_prediction = copy.deepcopy(model)
    optimizer_train = torch.optim.Adam(model_prediction.parameters(), lr=1e-03)

    for epoch in range(1, num_train_epochs):
        loss = train(model_prediction, optimizer_train, train_loader, loss_function, device)

        if epoch % 30 == 0 :
            val_acc = test(model_prediction, val_loader)
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')

    test_acc = test(model_prediction, test_loader)
    print('Accuracy on test set : {}' .format(test_acc))