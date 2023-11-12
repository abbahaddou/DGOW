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
import random
import numpy as np
import os
import wandb
from collections import Counter


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='DG', help='Config MG or DG')
parser.add_argument('--dataset', type=str, default='R8', help='Dataset string: R8, R52, OH, 20NGnew, MR')
parser.add_argument('--emb_type', type= str ,default = 'sdne', choices=['spectral','deepwalk',"sdne"] )
parser.add_argument('--remove_limit', type=int, default=20, help='Remove the words showing fewer than 2 times')
parser.add_argument('--min_sentences', type= int,default = 400 )
parser.add_argument('--window_size', type= int,default = 2)
parser.add_argument('--use_wandb', type= bool,default = False , choices=[True, False])
parser.add_argument('--embedding_file', type= str, help = "The folder containing the embeddings")

# Read Parser,
args = parser.parse_args()
print("dataset : ", args.dataset, " window size : " , args.window_size)


embedding_file = args.embedding_file



# read files & embeddings
word_id_map = np.load(os.path.join( embedding_file , 'word_id_map_{}_windowsise_{}_dataset_{}.npy'.format(args.config, args.window_size,args.dataset) ) , allow_pickle='TRUE').item()
# w = np.load(os.path.join( embedding_file  ,  'w_{}_windowsise_{}_dataset_{}.npy'.format(args.config, args.window_size,args.dataset) ))
v = np.load(os.path.join( embedding_file  , 'v_{}_windowsise_{}_dataset_{}.npy'.format(args.config, args.window_size,args.dataset)))

embedding = torch.tensor(v)

def get_sentence_emb(embedding , sentence) :
    sentence = [word_id_map[x] for x in sentence]
    word_embedding = [embedding[sentence[k],:].unsqueeze(0) for k in range(len(sentence))]
    sentence_emb = torch.concat(word_embedding, dim =0)
    sentence_emb = torch.sum(sentence_emb, 0)
    return sentence_emb

# Compute similarity

# Dataset
sentences, labels, train_size, test_size = get_dataset(args)
if args.emb_type == 'spectral':
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
set_labels = list(set(labels))
num_classes = len(set_labels)
train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]
train_labels = labels[:train_size]
test_labels = labels[train_size:]

tokenize_sentences, word_list = preprocess_data(train_sentences, test_sentences, args)

if args.config == 'DG' : 
    tokenize_sentences = []
    labels = []
    word_list = []
    
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
        labels_l = [l for i in range(len(tokenize_sentences_l))]
        labels  = labels + labels_l


# Choose the dimension of the embedding as the number of classes
if args.emb_type ==  'spectral' :
    #embedding = embedding[:,:num_classes]
    embedding = embedding[:,:64]

# Keep only significant labels
possible_labels = set(labels)
cluster_sentences = {}
for l in possible_labels :
    L = [ i_ for i_ in range(len(labels)) if labels[i_] == l]
    if len(L) > args.min_sentences :
        cluster_sentences[l] = random.sample(L, args.min_sentences)

significant_labels = list(cluster_sentences.keys())
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
label_pairs = []


for label_1 in significant_labels :
    for label_2 in significant_labels :
        if label_1 != label_2  and  ((label_1,label_2 ) not in label_pairs) and ((label_2,label_1 ) not in label_pairs ):
            all_sim = []
            L_1 = cluster_sentences[label_1]
            L_2 = cluster_sentences[label_2]

            for i in range(len(L_1)) :
                sentence_1 = tokenize_sentences[L_1[i]]
                sentence_emb_1 = get_sentence_emb(embedding , sentence_1)
                for j in range(len(L_2)) :
                    sentence_2 = tokenize_sentences[L_2[j]]
                    sentence_emb_2 = get_sentence_emb(embedding , sentence_2)
                    sim_value = cos(sentence_emb_1.unsqueeze(0), sentence_emb_2.unsqueeze(0))
                    #print(sim_value)
                    all_sim.append(sim_value.item())

            print("label_1 : ", label_1 , "  label_2  : " , label_2 , " :  ", np.mean(all_sim), '  +-  ', np.std(all_sim) )

            label_pairs.append((label_1,label_2 ))

for label_1 in significant_labels :

    label_2 = label_1
    all_sim = []
    L_1 = cluster_sentences[label_1]
    L_2 = cluster_sentences[label_2]

    for i in range(len(L_1)) :
        sentence_1 = tokenize_sentences[L_1[i]]
        sentence_emb_1 = get_sentence_emb(embedding , sentence_1)
        for j in range(len(L_1)) :
            if i != j :
                sentence_2 = tokenize_sentences[L_1[j]]
                sentence_emb_2 = get_sentence_emb(embedding , sentence_2)
                sim_value = cos(sentence_emb_1.unsqueeze(0), sentence_emb_2.unsqueeze(0))
                #print(sim_value, label_2 , label_1)
                all_sim.append(sim_value.item())

    print("label_1 : ", label_1 , "  label_2  : " , label_2 , " :  ", np.mean(all_sim) , '  +-   ', np.std(all_sim) )
    if args.use_wandb :
        wandb.log({"{}_{}_mean".format(label_1, label_2) : np.mean(all_sim), "{}_{}_std".format(label_1, label_2) : np.std(all_sim)})
