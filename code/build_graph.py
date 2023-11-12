# This code create a graph representing the corpus of each class

from tqdm.auto import tqdm
import scipy.sparse as sp
from math import log
import numpy as np
import sys


def ordered_word_pair(a, b):
    if a > b:
        return b, a
    else:
        return a, b

def get_adj(tokenize_sentences,word_id_map,word_list,args):
    window_size = args.window_size
    total_W = 0
    word_occurrence = {}
    word_pair_occurrence = {}

    node_size = len(word_list)

    def update_word_and_word_pair_occurrence(q):
        unique_q = list(set(q))
        for i in unique_q:
            try:
                word_occurrence[i] += 1
            except:
                word_occurrence[i] = 1
        for i in range(len(unique_q)):
            for j in range(i+1, len(unique_q)):
                word1 = unique_q[i]
                word2 = unique_q[j]
                word1, word2 = ordered_word_pair(word1, word2)
                try:
                    word_pair_occurrence[(word1, word2)] += 1
                except:
                    word_pair_occurrence[(word1, word2)] = 1
                


    for ind in range(len(tokenize_sentences)):

        words = tokenize_sentences[ind]
        q = []
        # push the first (window_size) words into a queue
        for i in range(min(window_size, len(words))):
            q += [word_id_map[words[i]]]
        # update the total number of the sliding windows
        total_W += 1
        # update the number of sliding windows that contain each word and word pair
        update_word_and_word_pair_occurrence(q)

        now_next_word_index = window_size
        # pop the first word out and let the next word in, keep doing this until the end of the document
        while now_next_word_index<len(words):
            q.pop(0)
            q += [word_id_map[words[now_next_word_index]]]
            now_next_word_index += 1
            # update the total number of the sliding windows
            total_W += 1
            # update the number of sliding windows that contain each word and word pair
            update_word_and_word_pair_occurrence(q)

    # calculate PMI for edges
    row = []
    col = []
    weight = []
    for word_pair in word_pair_occurrence:
        i = word_pair[0]
        j = word_pair[1]
        count = word_pair_occurrence[word_pair]
        word_freq_i = word_occurrence[i]
        word_freq_j = word_occurrence[j]
        pmi = log((count * total_W) / (word_freq_i * word_freq_j)) 
        if pmi <=0:
            continue
        row.append(i)
        col.append(j)
        weight.append(pmi)
        row.append(j)
        col.append(i)
        weight.append(pmi)
            
    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

    return adj     
