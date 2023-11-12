from build_dataset import get_dataset, get_restricted_dataset
from process import preprocess_data
import argparse
# Torch
import torch
import sys
import numpy as np
from utils import ordered_word_pair
from build_graph import get_adj
import networkx as nx
from scipy.sparse import block_diag,identity, csr_matrix
from spectral_embedding import *
from scipy import sparse, errstate, sqrt
import numpy as np
from collections import Counter
import operator


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='DG', help='Config MG or DG')
parser.add_argument('--dataset', type=str, default='MR', help='Dataset string: R8, R52, OH, 20NGnew, MR')
parser.add_argument('--remove_limit', type=int, default=2, help='Remove the words showing fewer than 2 times')
parser.add_argument('--window_size', type= int,default = 2 )
parser.add_argument('--embedding_file', type= str,default = "/home/yassine/Projects/Graphs_Text_Classification/final_code/similarity/embeddings/")
# Read Parser,
args = parser.parse_args()
print("dataset : ", args.dataset, " window size : " , args.window_size)

# Dataset
sentences, labels, train_size, test_size = get_dataset(args)

if args.dataset == 'R52' or args.dataset == '20NGnew' or args.dataset == 'OH' :
    restricted_num_classes = {'R52' : 8, 'OH':4 , '20NGnew' : 2}
    counter = dict(Counter(labels))
    keys = list(counter.keys())
    values = [counter[keys[i_]] for i_ in range(len(keys)) ]
    sort_index = list(np.argsort(values))
    sort_index.reverse()
    sort_index = sort_index[:restricted_num_classes[args.dataset]]
    restricted_classes  =set([keys[z] for z in sort_index]   )
    sentences, labels, train_size, test_size = get_restricted_dataset(args, restricted_classes)
    
#if args.dataset == 'MR' :
    #labels = labels[:int(len(labels)/2)]
    #sentences = sentences[:int(len(labels)/2)]
    #print(Counter(labels[:int(len(labels)/2)]))

train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]
train_labels = labels[:train_size]
test_labels = labels[train_size:]


if args.config == 'MG' :
    tokenize_sentences, word_list = preprocess_data(train_sentences, test_sentences, args)
    vocab_length = len(word_list)
    word_id_map = {}
    for i in range(vocab_length):
        word_id_map[word_list[i]] = i   

    # Graph construction
    adj = get_adj(tokenize_sentences,word_id_map,word_list,args.window_size)
    max_adj = csr_matrix.max(adj)
    min_adj = csr_matrix.min(adj)
    adj = (adj - min_adj) / (max_adj - min_adj)

elif args.config == 'DG' : 
    tokenize_sentences = []
    word_list = [len]
    set_labels = list(set(labels))
    for l in set_labels :
        # Filter the sentences belonging to the class cls
        indices_train = [i_ for i_ in range(len(train_labels)) if train_labels[i_]== l]
        indices_test = [i_ for i_ in range(len(test_labels)) if test_labels[i_]== l]
        train_sentences_l = [train_sentences[i] for i in indices_train]
        test_sentences_l = [test_sentences[i] for i in indices_test]
        tokenize_sentences_l, word_list_l = preprocess_data(train_sentences_l, test_sentences_l, args)
        tokenize_sentences_l = [[x + '_{}'.format(l) for x in tokenize_sentences_l[i_]] for i_ in range(len(tokenize_sentences_l))]
        word_list_l = [word_list_l[i_] + '_{}'.format(l) for i_ in range(len(word_list_l))] 
        tokenize_sentences = tokenize_sentences + tokenize_sentences_l
        word_list = word_list + word_list_l
    vocab_length = len(word_list)
    word_id_map = {}
    for i in range(vocab_length):
        word_id_map[word_list[i]] = i   

    # Graph construction
    adj = get_adj(tokenize_sentences,word_id_map,word_list,args.window_size)
    max_adj = csr_matrix.max(adj)
    min_adj = csr_matrix.min(adj)
    adj = (adj - min_adj) / (max_adj - min_adj)

# Save

np.save('word_id_map_{}_windowsise_{}_dataset_{}.npy'.format(args.config, args.window_size,args.dataset), word_id_map) 


degrees = adj.dot(np.ones(vocab_length))
degree_matrix = sparse.diags(degrees, format = 'csr')
laplacian = degree_matrix - adj
degrees_inv_sqrt = 1.0 / sqrt(degrees)
degrees_inv_sqrt[isinf(degrees_inv_sqrt)] = 0
weight_matrix = sparse.diags(degrees_inv_sqrt, format = 'csr')
laplacian = weight_matrix.dot(laplacian.dot(weight_matrix))
w, v = np.linalg.eigh(laplacian.todense())

np.save('w_{}_windowsise_{}_dataset_{}.npy'.format(args.config, args.window_size,args.dataset), w) 
np.save('v_{}_windowsise_{}_dataset_{}.npy'.format(args.config, args.window_size,args.dataset), v) 
