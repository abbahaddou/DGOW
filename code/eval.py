
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score
import torch

def eval_model(test_tokenize_sentences,encoded_test_labels,vocab_length, num_class, word_id_map, gnn,sequential_model , agg,sp_feature ,edge_index, edge_weight) :
    
    # Evaluation the model
    gnn.eval()
    sequential_model.eval()
    agg.eval()

    ground_thruts = []
    predictions = []

    encoder_emb = gnn(sp_feature.float() ,edge_index, edge_weight.float()).detach()

    for i in tqdm(range(len(test_tokenize_sentences))) :

        sentence = test_tokenize_sentences[i]
        sentence = [word_id_map[sentence[j]]  for j in range(len(sentence)) ]

        # The node of words depends on the classes since each word has num_classes node representations
        dict_cls = {}
        for cls in range(num_class) :
            
            sentence_in_cls = [cls*vocab_length + sentence[i] for i in range(len(sentence))]
            encoder_emb_i_s = encoder_emb.unsqueeze(0)[:,sentence_in_cls,:].detach()
            decoder_emb_i_s = sequential_model(encoder_emb_i_s)[0].detach()  # Output of the LSTM is output, (hn, cn)  . we take only hn that store the last hidden state of each element in the sequence
            predicted_prob = agg(decoder_emb_i_s)
            dict_cls[cls] = predicted_prob.item()

        keys = list(dict_cls.keys())
        v = [dict_cls[k] for k in keys ]
        #print( keys[v.index(max(v))], encoded_test_labels[i], max(v))
        predictions.append( keys[v.index(max(v))] )
        ground_thruts.append( encoded_test_labels[i] )
    
    print("predictions")
    print(Counter(predictions))
    print("ground_thruts")
    print(Counter(ground_thruts))
    test_accuracy = accuracy_score(ground_thruts, predictions)
    return test_accuracy

