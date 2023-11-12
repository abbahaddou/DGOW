import pandas as pd
import os

# 
def get_dataset(args):
    dataset = args.dataset

    df = pd.read_csv(os.path.join( "data" , "{}.csv".format(dataset)))
    
    sentences = df['text'].tolist()
    labels = df['label'].tolist()
    train_or_test = df['train'].tolist()

    train_sentences, train_labels, test_sentences, test_labels = [],[],[],[]
    train_size, test_size = 0, 0
    for i in range(len(train_or_test)):
        if train_or_test[i] == 'train':
            train_sentences.append(sentences[i])
            train_labels.append(labels[i])
            train_size += 1
        elif train_or_test[i] == 'test':
            test_sentences.append(sentences[i])
            test_labels.append(labels[i])
            test_size += 1

    return train_sentences+test_sentences, train_labels+test_labels, train_size, test_size



def get_restricted_dataset(args, restricted_classes):
    dataset = args.dataset

    df = pd.read_csv(os.path.join( "data" , "{}.csv".format(dataset)))
    
    sentences = df['text'].tolist()
    labels = df['label'].tolist()
    train_or_test = df['train'].tolist()

    train_sentences, train_labels, test_sentences, test_labels = [],[],[],[]
    train_size, test_size = 0, 0
    for i in range(len(train_or_test)):
        if labels[i] in restricted_classes:
            if train_or_test[i] == 'train':
                train_sentences.append(sentences[i])
                train_labels.append(labels[i])
                train_size += 1
            elif train_or_test[i] == 'test':
                test_sentences.append(sentences[i])
                test_labels.append(labels[i])
                test_size += 1

    return train_sentences+test_sentences, train_labels+test_labels, train_size, test_size