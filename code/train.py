import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os
from collections import Counter
import datetime
import numpy as np
from torch.optim.lr_scheduler import StepLR
from utils import divide_chunks
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
from eval import eval_model 


def train_model(train_tokenize_sentences,encoded_train_labels,val_tokenize_sentences,encoded_val_labels,test_tokenize_sentences,encoded_test_labels, sp_feature ,edge_index, edge_weight, word_id_map ,gnn,sequential_model , agg, epochs, device,vocab_length , num_class,  args) :

        counter = dict(Counter(encoded_train_labels))
        keys = list(counter.keys())
        class_card  = [counter[keys[i_]] for i_ in range(len(keys)) ]
        # To delete
        def tokenize_sentence(sent) :
            sent = np.array([word_id_map[sent[i]] for i in range(len(sent))])
            return sent

        encoded_val_labels = np.array(encoded_val_labels)

        all_val_acc   = [0,0,0,0]
        loss_fn = nn.BCELoss()
        opt_gen = optim.Adam(list(gnn.parameters()) + list(sequential_model.parameters()) + list(agg.parameters()) , lr=args.lr)
        training_step = 0 
        best_gnn = gnn
        best_sequential_model = sequential_model
        best_agg = agg
        best_val_acc = 0
        for epoch in range(epochs) :

            sentences = train_tokenize_sentences + train_tokenize_sentences
            labels = encoded_train_labels
            if args.neg_sampling == 'ferq_based' :
                for t_ in range(len(train_tokenize_sentences)) :
                    labels.append( random.choices(population=[keys[z_] for z_ in range(len(keys)) if keys[z_] !=  encoded_train_labels[t_]], weights=[class_card[z_] for z_ in range(len(keys)) if keys[z_] !=  encoded_train_labels[t_]])[0]   )
            elif args.neg_sampling == 'uniform' :
                for t_ in range(len(train_tokenize_sentences)) :
                    labels.append( (encoded_train_labels[t_] + random.randint(1,num_class-1)) % num_class ) 
            
            labels = np.array(labels)
            ground_thruth = [1 for t_ in range(len(train_tokenize_sentences))]  + [0 for t_ in range(len(train_tokenize_sentences))] 
            ground_thruth = torch.tensor(ground_thruth).to(device)
            
            l = list(np.arange(len(sentences)))
            random.shuffle(l)
            batches = divide_chunks(l,args.batch_size)
            batches.pop() # remove the last batch since it size < batch size
            
            for batch in tqdm(batches) :
                
                # the  batch of sentences
                ground_thruth_batch = ground_thruth[batch]
                label_batch = labels[batch]
                sentences_batch = [sentences[y] for y in batch]
                sentences_batch = [tokenize_sentence(sentences_batch[i_]) for i_ in range(len(sentences_batch)) ]
                sentences_batch = [label_batch[j_]*vocab_length + sentences_batch[j_]  for j_ in range(len(sentences_batch)) ]

                # Refine the embedding using GNN
                prdicted_labels = torch.tensor([]).to(device)
                encoder_emb = gnn(sp_feature.float() ,edge_index, edge_weight.float())
                
                for i_s ,  sent in enumerate(batch) :
                    encoder_emb_i_s = encoder_emb.unsqueeze(0)[:,sentences_batch[i_s],:]
                    decoder_emb_i_s = sequential_model(encoder_emb_i_s)[0]
                    predicted_prob_i_s = agg(decoder_emb_i_s)
                    prdicted_labels = torch.cat([prdicted_labels, predicted_prob_i_s] )
                
                loss  = loss_fn(prdicted_labels.float() , ground_thruth_batch.float())
                opt_gen.zero_grad()
                loss.backward()
                opt_gen.step()

                training_step += 1

                ## validation
                if training_step % 40 == 0 :
                    with torch.no_grad() :
                        val_acc = eval_model(val_tokenize_sentences,encoded_val_labels,vocab_length, num_class, word_id_map, gnn,sequential_model , agg,sp_feature ,edge_index, edge_weight) 
                        # val_acc = eval_model(test_tokenize_sentences[:20],encoded_test_labels[:20],vocab_length, num_class, word_id_map, gnn,sequential_model , agg,sp_feature ,edge_index, edge_weight) 
                        if best_val_acc < val_acc : 
                            best_gnn = gnn
                            best_sequential_model = sequential_model
                            best_agg = agg
                            best_val_acc = val_acc
                    if val_acc < min(all_val_acc[-4:]) :
                         return best_gnn,best_sequential_model , best_agg
                    all_val_acc.append(val_acc)
                    gnn.train()
                    sequential_model.train()
                    agg.train()
        return best_gnn,best_sequential_model , best_agg