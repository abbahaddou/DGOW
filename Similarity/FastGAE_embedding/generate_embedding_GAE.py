from build_dataset import get_dataset, get_restricted_dataset
from process import preprocess_data
import argparse
# Torch
import torch
import sys
import numpy as np
from build_graph import get_adj
import networkx as nx
from scipy.sparse import block_diag,identity, csr_matrix
from scipy import sparse, errstate, sqrt
import numpy as np
from collections import Counter
import operator
from train_fastGAE import train

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='DG', help='Config DG or MG')
parser.add_argument('--dataset', type=str, default='R8', help='Dataset string: R8, R52, OH, 20NGnew, MR')
parser.add_argument('--remove_limit', type=int, default=2, help='Remove the words showing fewer than 2 times')
parser.add_argument('--window_size', type= int,default = 2 )
parser.add_argument('--min_sentences', type= int,default = 400 )
parser.add_argument('--embedding_file', type= str, help="The folder to store the generated embeddings")
# Read Parser,
args = parser.parse_args()
print("dataset : ", args.dataset, " window size : " , args.window_size)

# Dataset
sentences, labels, train_size, test_size = get_dataset(args)
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
    
    g_nx = nx.from_scipy_sparse_matrix(adj)
    final_emb = train(g_nx)
    v = final_emb.detach().numpy()



elif args.config == 'DG' : 
    tokenize_sentences = []
    word_list = []
    set_labels = list(set(labels))
    # Keep only significant labels
    all_labels = set(labels)
    #significant_labels = []
    #for l in all_labels :
    #    L = [ i_ for i_ in range(len(labels)) if labels[i_] == l]
    #    if len(L) > args.min_sentences :
    #        significant_labels.append(l)
    
    for q_ , l in enumerate(all_labels) :
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
        
        vocab_length_l = len(word_list_l)
        word_id_map_l = {}
        for i in range(vocab_length_l):
            word_id_map_l[word_list_l[i]] = i 
        sdne_mask = csr_matrix(np.ones((vocab_length_l,vocab_length_l))) 
        
        if q_ == 0  :
            full_sdne_mask = sdne_mask
        else :      
            full_sdne_mask = block_diag([full_sdne_mask, sdne_mask])
    vocab_length = len(word_list)
    word_id_map = {}
    for i in range(vocab_length):
        word_id_map[word_list[i]] = i  
     
    # Graph construction
    adj = get_adj(tokenize_sentences,word_id_map,word_list,args.window_size)
    max_adj = csr_matrix.max(adj)
    min_adj = csr_matrix.min(adj)
    adj = (adj - min_adj) / (max_adj - min_adj)
    # Graph construction
    g_nx = nx.from_scipy_sparse_matrix(adj)
    final_emb = train(g_nx)
    v = final_emb.detach().numpy()

# Save
np.save('word_id_map_{}_windowsise_{}_dataset_{}.npy'.format(args.config, args.window_size,args.dataset), word_id_map) 
np.save('v_{}_windowsise_{}_dataset_{}.npy'.format(args.config, args.window_size,args.dataset), v) 
