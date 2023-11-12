# Packages 
import argparse
from inspect import getinnerframes
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os
from collections import Counter
import datetime
import numpy as np
from torch.optim.lr_scheduler import StepLR

from scipy.sparse import block_diag,identity, csr_matrix
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.utils.convert import from_scipy_sparse_matrix #  To adapt the adjacency matrix with torch_geometric
import sys
from eval import eval_model
from utils import decide_device, sparse_mx_to_torch_sparse_tensor, normalize_adj, divide_chunks
from build_dataset import get_dataset
from process import encode_labels, preprocess_data
from models import GCN, GAT, GATV2,GIN, PNA, GraphSAGE, Cheb  
from models import LSTM_w_activations, MLP_agg  , Sum_agg,Prod_agg, Perceptron, AVG_agg
from build_graph import get_adj
from train import train_model
import os

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IMDB', help='Dataset string: R8, R52, OH, 20NGnew, MR')
parser.add_argument('--setting', type=str, default='transductive',  choices=['transductive' , 'inductive'])
parser.add_argument('--gnn', type=str, default='gcn', help='The architechture of GNN', choices=['gcn' , 'gat', 'gatv2','cheb','gin' ,'graphsage'])   # 'gin' and 'graphsage' support only dense features ==> CUDA MEMORY
parser.add_argument('--agg_type', type=str, default='avg', help='The architechture of aggregation function', choices=['mlp' , 'avg', 'prod' , 'sum']) 
parser.add_argument('--remove_limit', type=int, default=2, help='Remove the words showing fewer than 2 times')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--use_gpu', type=int, default=1, help='Whether to use GPU, 1 means True and 0 means False. If True and no GPU available, will use CPU instead.')
parser.add_argument('--word_embedding',type = str, default = "one_hot" ,choices= ["one_hot" , "glove.6B.50d","glove.6B.100d","glove.6B.200d","glove.6B.300d"], help="The features of the word in the vocabulary")
parser.add_argument('--window_size',type = int, default = 2, help="The window size")
parser.add_argument('--gnn_layers',type = int, default = 4 , help="The nummber of layers in the GNN")
parser.add_argument('--use_wandb', type= bool,default = False , choices=[True, False])
parser.add_argument('--batch_size', type= int,default = 64)
parser.add_argument('--lstm_layers', type= int,default = 2)
parser.add_argument('--neg_sampling', type= str,default = 'ferq_based' , choices=['uniform','ferq_based' ])
parser.add_argument('--lr', type= float,default = 0.001 )

# Read Parser,
args = parser.parse_args()
# Device
device = decide_device(args)

# Get dataset
sentences, labels, train_size, test_size = get_dataset(args)

# split train/test sentences and labels
train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]
train_labels = labels[:train_size]
test_labels = labels[train_size:]

# Preprocess text and labels
labels, num_class, lEnc = encode_labels(train_labels, test_labels, args) # Encode labels into integers starting from 0
tokenize_sentences, word_list = preprocess_data(train_sentences, test_sentences, args)

train_tokenize_sentences = tokenize_sentences[:train_size]
test_tokenize_sentences = tokenize_sentences[train_size:]

# Generate Graph for each classes
encoded_train_labels = lEnc.transform(train_labels)
encoded_test_labels = lEnc.transform(test_labels)

# Check if it exsit an empty sentences after data processing
original_train_size = len(train_tokenize_sentences)
empty_index = [i_ for i_ in range(len(train_tokenize_sentences)) if train_tokenize_sentences[i_] != [] ]
train_tokenize_sentences = [train_tokenize_sentences[i_] for i_ in range(original_train_size) if i_ in empty_index]
encoded_train_labels = np.array([encoded_train_labels[i_] for i_ in range(original_train_size) if i_ in empty_index])

idx_train , idx_val = train_test_split(list(np.arange(len(train_tokenize_sentences))), shuffle=True,test_size=0.1, train_size=0.9)
val_tokenize_sentences = [train_tokenize_sentences[i_] for i_ in  idx_val]
train_tokenize_sentences = [train_tokenize_sentences[i_] for i_ in  idx_train]
encoded_val_labels = [encoded_train_labels[i_] for i_ in  idx_val]
encoded_train_labels = [encoded_train_labels[i_] for i_ in  idx_train]

vocab_length = len(word_list)
word_id_map = {}
for i in range(vocab_length):
    word_id_map[word_list[i]] = i


# Features 
if args.word_embedding == 'one_hot' :
    input_size = vocab_length  * num_class
elif args.word_embedding == 'glove.6B.50d' :
    input_size = 50 
elif args.word_embedding == 'glove.6B.100d' :
    input_size = 100 
elif args.word_embedding == 'glove.6B.200d' :
    input_size = 200 
elif args.word_embedding == 'glove.6B.300d' :
    input_size = 300 


# Identity matrix
if args.word_embedding == 'one_hot' :
    sp_feature = sparse_mx_to_torch_sparse_tensor(identity(vocab_length * num_class)).to(device)

elif  args.word_embedding in ["glove.6B.50d","glove.6B.100d","glove.6B.200d","glove.6B.300d"]:
    embeddings = {}
    with open(os.path.join(os.getcwd(), "glove_emb" , "{}.txt".format(args.word_embedding)),'rt') as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        embeddings[i_word] = torch.tensor(i_embeddings).unsqueeze(0)
    all_embeddings = torch.concat(list(embeddings.values()), 0)
    unk_embeding = torch.mean(all_embeddings, 0).unsqueeze(0)
    vocab_glove = list(embeddings.keys())
    
    sp_feature =[]
    for i in range(vocab_length):
        if word_list[i] in vocab_glove :
            sp_feature.append(embeddings[word_list[i]])
        else :
            sp_feature.append(unk_embeding)

    sp_feature = torch.concat(sp_feature, 0)
    # Each word has num_class representation ==> duplacte 
    sp_feature = torch.concat([sp_feature for i_ in range(num_class)], 0).to(device)


# Adjacency Matrix
for cls in tqdm(range(num_class)) :
    # Filter the sentences belonging to the class cls
    #if args.setting == 'transductive' :
    #    filtered_cls = [train_tokenize_sentences[i_] for i_ in range(len(encoded_train_labels)) if encoded_train_labels[i_]== cls] + [val_tokenize_sentences[i_] for i_ in range(len(encoded_val_labels)) if encoded_val_labels[i_]== cls] + [test_tokenize_sentences[i_] for i_ in range(len(encoded_test_labels)) if encoded_test_labels[i_]== cls]
    #else : 
    #    filtered_cls = [train_tokenize_sentences[i_] for i_ in range(len(encoded_train_labels)) if encoded_train_labels[i_]== cls]
    filtered_cls = [train_tokenize_sentences[i_] for i_ in range(len(encoded_train_labels)) if encoded_train_labels[i_]== cls]
    # Create the graph of the class (weighted using PMI), and the embedding of words.
    # No doc node in the graph since consider the text classification as a path classification problem
    adj_cls = get_adj(filtered_cls,word_id_map,word_list,args)
    #adj_cls, norm_item = normalize_adj(adj_cls + sp.eye(adj_cls.shape[0]))
    if cls == 0 :
        full_adj_sp = adj_cls
    else :
        full_adj_sp = block_diag([full_adj_sp, adj_cls])


# Make the value of the adj matrix between 0 and 1
full_adj_sp = (full_adj_sp - csr_matrix.min(full_adj_sp))/(csr_matrix.max(full_adj_sp) - csr_matrix.min(full_adj_sp))

normalize_full_adj_sp, norm_item = normalize_adj(full_adj_sp + identity(num_class * vocab_length)  )

# Convert the adj to sparse torch tensorsample_node_num
full_adj = sparse_mx_to_torch_sparse_tensor(full_adj_sp).to(device)
edge_index, edge_weight = from_scipy_sparse_matrix(full_adj_sp)
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)

#normalize_full_adj= sparse_mx_to_torch_sparse_tensor(normalize_full_adj_sp).to(device)

all_acc = []

for tt in range(4 ) :
    # Models
    if args.gnn == 'gcn' :
        gnn = GCN(gnn_layers = args.gnn_layers , in_channels =  input_size, hidden_channels = 256, out_channels =  256).to(device)
    elif args.gnn == 'gat' :
        gnn = GAT(gnn_layers = args.gnn_layers , in_channels =  input_size, hidden_channels = 256, out_channels =  256).to(device)
    elif args.gnn == 'gatv2' :
        gnn = GATV2(gnn_layers = args.gnn_layers , in_channels =  input_size, hidden_channels = 256, out_channels =  256).to(device)
    elif args.gnn == 'gin' :
        gnn = GIN(gnn_layers = args.gnn_layers , in_channels =  input_size, hidden_channels = 256, out_channels =  256).to(device)
    elif args.gnn == 'pna' :
        gnn = PNA(gnn_layers = args.gnn_layers , in_channels =  input_size, hidden_channels = 256, out_channels =  256).to(device)
    elif args.gnn == 'graphsage' :
        gnn = GraphSAGE(gnn_layers = args.gnn_layers , in_channels =  input_size, hidden_channels = 256, out_channels =  256).to(device)
    elif args.gnn == 'cheb' :
        gnn = Cheb(gnn_layers = args.gnn_layers , in_channels =  input_size, hidden_channels = 256, out_channels =  256).to(device)

    sequential_model = decoder = LSTM_w_activations(input_size =256 , hidden_size = 256, num_layers = args.lstm_layers).to(device)

    if args.agg_type == 'mlp' :
        agg =  MLP_agg().to(device)
    elif args.agg_type == 'avg': 
        agg =  AVG_agg().to(device)
    elif args.agg_type == 'sum': 
        agg =  Sum_agg().to(device)
    elif args.agg_type == 'prod': 
        agg =  Prod_agg(args.window_size).to(device)

    gnn,sequential_model , agg = train_model(train_tokenize_sentences,encoded_train_labels,val_tokenize_sentences,encoded_val_labels,test_tokenize_sentences,encoded_test_labels, sp_feature ,edge_index, edge_weight, word_id_map ,gnn,sequential_model , agg, args.epochs, device,vocab_length , num_class,  args)
    test_acc = eval_model(test_tokenize_sentences,encoded_test_labels,vocab_length, num_class, word_id_map, gnn,sequential_model , agg,sp_feature ,edge_index, edge_weight) 
    print("test_acc" , test_acc)
    all_acc.append(test_acc)
