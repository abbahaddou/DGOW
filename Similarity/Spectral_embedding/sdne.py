import torch
import numpy as np
import scipy.sparse as sparse
import sys
import random


def process_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx

class Regularization(torch.nn.Module):

    def __init__(self, model, gamma=0.01, p=2, device="cpu"):
        super().__init__()
        if gamma <= 0:
            print("param weight_decay can not be <= 0")
            exit(0)
        self.model = model
        self.gamma = gamma
        self.p = p
        self.device = device
        self.weight_list = self.get_weight_list(model) # 取出参数的列表
        self.weight_info = self.get_weight_info(self.weight_list) # 打印参数的信息

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, model):
        self.weight_list = self.get_weight_list(model)
        reg_loss = self.regulation_loss(self.weight_list, self.gamma, self.p)
        return reg_loss

    def regulation_loss(self, weight_list, gamma, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss += l2_reg
        reg_loss = reg_loss * gamma
        return reg_loss

    def get_weight_list(self, model):
        weight_list = []
        # 返回参数的名字和参数自己
        for name, param in model.named_parameters():
            # 这里只取weight 未取bias
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def get_weight_info(self, weight_list):
        # 打印被正则化的参数的名称
        print("#"*10, "regulations weight", "#"*10)
        for name, param in weight_list:
            print(name)
        print("#"*25)
        
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score




class GraphBaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def fit(self):
        pass


class TopKRanker(OneVsRestClassifier):

    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return np.asarray(all_labels)

class MultiClassifier(object):


    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer()

    def fit(self, X, y, y_all):
        '''
        :param X:
        :param y:
        :param y_all: 所有的标签
        :return:
        '''
        self.binarizer.fit(y_all)
        X_train = [self.embeddings[x] for x in X]
        y_train = self.binarizer.transform(y)
        self.clf.fit(X_train, y_train)

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        y_pred = self.clf.predict(X_, top_k_list=top_k_list)
        return y_pred

    def evaluate(self, X, y):
        top_k_list = [len(l) for l in y]
        y_pred = self.predict(X, top_k_list)
        y = self.binarizer.transform(y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(y, y_pred, average=average)
        results['acc'] = accuracy_score(y, y_pred)
        print('-------------------')
        print(results)
        print('-------------------')
        return results

    def evaluate_hold_out(self, X, y, test_size=0.2, random_state=123):
        np.random.seed(random_state)
        train_size = int((1-test_size) * len(X))
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(train_size)]
        y_train = [y[shuffle_indices[i]] for i in range(train_size)]
        X_test = [X[shuffle_indices[i]] for i in range(train_size, len(X))]
        y_test = [y[shuffle_indices[i]] for i in range(train_size, len(X))]

        self.fit(X_train, y_train, y)

        return self.evaluate(X_test, y_test)


class SDNEModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_layers, alpha, beta, device="cpu"):
        super(SDNEModel, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        input_dim_copy = input_dim
        layers = []
        for layer_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            input_dim = layer_dim
        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for layer_dim in reversed(hidden_layers[:-1]):
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            input_dim = layer_dim
        layers.append(torch.nn.Linear(input_dim, input_dim_copy))
        layers.append(torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(*layers)
    def forward(self, A, L,mask):

        Y = self.encoder(A)
        A_hat = self.decoder(Y)
        # loss_2nd 二阶相似度损失函数
        beta_matrix = torch.ones_like(A)
        mask = A != 0
        beta_matrix[mask] = self.beta
        loss_2nd =  torch.sum(mask  *   torch.pow((A - A_hat) * beta_matrix, 2)) / torch.sum(mask)
        loss_1st =  self.alpha * 2 * torch.trace(torch.matmul(torch.matmul(Y.transpose(0,1), L), Y))
        print(loss_1st, loss_2nd)
        return loss_2nd + loss_1st




class SDNE(GraphBaseModel):

    def __init__(self, graph,full_sdne_mask, hidden_layers=None, alpha=1e-5, beta=5, gamma=1e-5, device="cpu"):
        super().__init__()
        self.graph = graph
        self.idx2node, self.node2idx = process_nxgraph(graph)
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.sdne = SDNEModel(self.node_size, hidden_layers, alpha, beta)
        self.device = device
        self.embeddings = {}
        self.gamma = gamma
        self.full_sdne_mask = full_sdne_mask
        self.full_sdne_mask = torch.from_numpy(self.full_sdne_mask.toarray()).float().to(self.device)

        adjacency_matrix, laplace_matrix = self.__create_adjacency_laplace_matrix()

        self.adjacency_matrix = torch.from_numpy(adjacency_matrix.toarray()).float().to(self.device)
        self.laplace_matrix = torch.from_numpy(laplace_matrix.toarray()).float().to(self.device)


    def fit(self, batch_size=512, epochs=1, initial_epoch=0, verbose=1):
        num_samples = self.node_size
        self.sdne.to(self.device)
        optimizer = torch.optim.Adam(self.sdne.parameters())
        if self.gamma:
            regularization = Regularization(self.sdne, gamma=self.gamma)
        if batch_size >= self.node_size:
            batch_size = self.node_size
            print('batch_size({0}) > node_size({1}),set batch_size = {1}'.format(
                batch_size, self.node_size))
            for epoch in range(initial_epoch, epochs):
                loss_epoch = 0
                optimizer.zero_grad()
                loss = self.sdne(self.adjacency_matrix, self.laplace_matrix , self.full_sdne_mask)
                if self.gamma:
                    reg_loss = regularization(self.sdne)
                    # print("reg_loss:", reg_loss.item(), reg_loss.requires_grad)
                    loss = loss + reg_loss
                loss_epoch += loss.item()
                loss.backward()
                optimizer.step()
                if verbose > 0:
                    print('Epoch {0}, loss {1} . >>> Epoch {2}/{3}'.format(epoch + 1, round(loss_epoch / num_samples, 4), epoch+1, epochs))
        else:
            steps_per_epoch = (self.node_size - 1) // batch_size + 1
            for epoch in range(initial_epoch, epochs):
                loss_epoch = 0
                for i in range(steps_per_epoch):
                    idx = np.arange(i * batch_size, min((i+1) * batch_size, self.node_size))
                    random.shuffle(idx)

                    mask_train = self.full_sdne_mask[idx, :]
                    A_train = self.adjacency_matrix[idx, :]
                    L_train = self.laplace_matrix[idx][:,idx]

                    optimizer.zero_grad()
                    loss = self.sdne(A_train, L_train,mask_train)
                    loss_epoch += loss.item()
                    loss.backward()
                    optimizer.step()

                if verbose > 0:
                    print('Epoch {0}, loss {1} . >>> Epoch {2}/{3}'.format(epoch + 1, round(loss_epoch / num_samples, 4),
                                                                         epoch + 1, epochs))

    def get_embeddings(self):
        if not self.embeddings:
            self.__get_embeddings()
        embeddings = self.embeddings
        return embeddings

    def __get_embeddings(self):
        embeddings = {}
        with torch.no_grad():
            self.sdne.eval()
            embed = self.sdne.encoder(self.adjacency_matrix)
            for i, embedding in enumerate(embed.numpy()):
                embeddings[self.idx2node[i]] = embedding
        self.embeddings = embeddings


    def __create_adjacency_laplace_matrix(self):
        node_size = self.node_size
        node2idx = self.node2idx
        adjacency_matrix_data = []
        adjacency_matrix_row_index = []
        adjacency_matrix_col_index = []
        for edge in self.graph.edges():
            v1, v2 = edge
            edge_weight = self.graph[v1][v2].get("weight", 1.0)
            adjacency_matrix_data.append(edge_weight)
            adjacency_matrix_row_index.append(node2idx[v1])
            adjacency_matrix_col_index.append(node2idx[v2])
        adjacency_matrix = sparse.csr_matrix((adjacency_matrix_data,
                                              (adjacency_matrix_row_index, adjacency_matrix_col_index)),
                                             shape=(node_size, node_size))
        adjacency_matrix_ = sparse.csr_matrix((adjacency_matrix_data+adjacency_matrix_data,
                                               (adjacency_matrix_row_index+adjacency_matrix_col_index,
                                                adjacency_matrix_col_index+adjacency_matrix_row_index)),
                                              shape=(node_size, node_size))
        degree_matrix = sparse.diags(adjacency_matrix_.sum(axis=1).flatten().tolist()[0])
        laplace_matrix = degree_matrix - adjacency_matrix_
        return adjacency_matrix, laplace_matrix

