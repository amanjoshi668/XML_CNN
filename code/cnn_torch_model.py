import os
import pickle
import random
import numpy as np
import scipy.sparse as sp
from w2v import load_word2vec

import torch
from torch import nn
def out_size(l_in, kernel_size, padding=0, dilation=1, stride=1):
    a = l_in + 2*padding - dilation*(kernel_size - 1) - 1
    b = int(a/stride)
    return b + 1
class CNN_model(nn.Module):
    def __init__(self, args):
        # Model Hyperparameters
        super(CNN_model, self).__init__()
        self.sequence_length = args.sequence_length
        self.embedding_dim = args.embedding_dim
        self.filter_sizes = args.filter_sizes
        self.num_filters = args.num_filters
        self.pooling_units = args.pooling_units
        self.pooling_type = args.pooling_type
        self.hidden_dims = args.hidden_dims

        # Training Hyperparameters
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs

        # Model variation and w2v pretrain type
        self.model_variation = args.model_variation        # CNN-rand | CNN-pretrain
        self.pretrain_type = args.pretrain_type

        # Fix random seed
        np.random.seed(1126)
        random.seed(1126)
        # self.rng = torch.shared_randomstreams.RandomStreams(seed=1126)

        # Optimizer

        optimizer = torch.optim.Adam(self.parameters(), lr = 0.01)

        # Model
        self.dropout1 = nn.Dropout(p = 0.25)
        self._final_in_shape = 0
        self.convs = []
        for fsz in self.filter_sizes:
            l_conv = nn.Conv1d(in_channels = self.embedding_dim, out_channels=self.num_filters, kernel_size=fsz, stride=2)
            l_out_size = out_size(self.sequence_length, fsz, stride=2)
            pool_size = l_out_size // self.pooling_units
            if self.pooling_type == 'average':
                l_pool = nn.AvgPool1d(pool_size, stride=None, count_include_pad=True)
                pool_out_size = (int((l_out_size - pool_size)/pool_size) + 1)*self.num_filters
            elif self.pooling_type == 'max':
                l_pool = nn.MaxPool1d(2, stride=1)
                pool_out_size = (int(l_out_size*self.num_filters - 2) + 1)
            else:
                raise NotImplementedError('Unknown pooling layer')
            self._final_in_shape += pool_out_size
            self.convs.append((l_conv, l_pool))
        self.hidden = nn.Linear(in_features=self._final_in_shape, out_features=self.hidden_dims)
        self.dropout2 = nn.Dropout(p = 0.5)


    def add_data(self, X_trn, Y_trn):
        """
            add data to the model.
        """
        self.X_trn = X_trn
        self.Y_trn = Y_trn
        self.output_dim = Y_trn.shape[1]
        self.out_layer = nn.Linear(in_features=self.hidden_dims, out_features=self.output_dim)

    def add_pretrain(self, vocabulary, vocabulary_inv):
        print ('model_variaton:', self.model_variation)
        if self.model_variation=='pretrain':
            embedding_weights = load_word2vec(self.pretrain_type, vocabulary_inv, self.embedding_dim)
        elif self.model_variation=='random':
            embedding_weights = None
        else:
            raise ValueError('Unknown model variation')
        self.embedding_weights = embedding_weights
        self.vocab_size = len(vocabulary)

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        if self.model_variation == "pretrain":
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights).type(torch.FloatTensor))
            self.embedding.weight.requires_grad=False
        elif self.model_variation != 'random':
            raise NotImplementedError('Unknown model_variation')

    def forward(self, X):
        X = self.embedding(X)
        # X = X.unsqueeze(1)
        X = self.dropout1(X)
        X = X.permute(0,2,1)
        # print("Shape after Embedding: ", X.shape)
        conv_out = []
        for (conv, pool) in self.convs:
            x = conv(X)
            x = x.view(x.shape[0], 1, x.shape[1]*x.shape[2])
            x = pool(x)
            x = x.view(x.shape[0],-1)
            conv_out.append(x)
        # X = [torch.relu(conv(X)) for (conv, pool) in self.convs]
        # X = [X.view(x.shape[0], 1, x.shape[1]*x.shape[2]) for x in X]
        # X = [x.view(x.shape[0],-1) for x in X]
        if len(self.filter_sizes)>1:
            X = torch.cat(conv_out,1)
        else:
            X = conv_out[0]
        # X = torch.cat(X, 1)
        # print(X.shape)
        X = torch.relu(self.hidden(X))#.view(-1, self._final_in_shape)))
        X = self.dropout2(X)
        X = torch.sigmoid(self.out_layer(X))
        return X

    def fit(self, lr=0.01):
        nr_trn_num = self.X_trn.shape[0]
        nr_batches = int(np.ceil(nr_trn_num / float(self.batch_size)))
        #nr_batches = nr_trn_num // self.batch_size
        self.train()
        criterion = nn.BCELoss()
        trn_loss = []
        for batch_idx in np.random.permutation(range(nr_batches)):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, nr_trn_num)
            #end_idx = (batch_idx + 1) * self.batch_size
            X = torch.from_numpy(self.X_trn[start_idx:end_idx]).type(torch.LongTensor)
            Y = torch.from_numpy(self.Y_trn[start_idx:end_idx].toarray()).type(torch.FloatTensor)
            Y_pred = self.forward(X).float()
            loss = criterion(Y_pred, Y)
            optimizer.zero_grad()
            loss.backward() # Backpropagation
            optimizer.step() 
            trn_loss.append(loss.item)
        return np.mean(loss)
    
    def predict(self, X_tst=None, batch_size = 8192, top_k=50, max_k=None):
        self.eval()
        nr_tst_num = X_tst.shape[0]
        nr_batches = int(np.ceil(nr_tst_num / float(batch_size)))
        row_idx_list, col_idx_list, val_idx_list = [], [], []
        #X_pred = np.zeros((X_tst.shape[0], self.hidden_dims))
        for batch_idx in range(nr_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, nr_tst_num)
            Y_pred = self.forward(torch.from_numpy(X_tst[start_idx:end_idx]).type(torch.LongTensor))
            #X_pred[start_idx:end_idx, :] = X_hidden
            for i in range(Y_pred.shape[0]):
                sorted_idx = np.argpartition(-Y_pred[i, :], top_k)[:top_k]
                row_idx_list += [i + start_idx] * top_k
                col_idx_list += (sorted_idx).tolist()
                val_idx_list += Y_pred[i, sorted_idx].tolist()
        m = max(row_idx_list) + 1
        n = max_k
        Y_pred = sp.csr_matrix((val_idx_list, (row_idx_list, col_idx_list)), shape=(m, n))
        return Y_pred

    # def load_model(model, name, optimizer=None):
    # if(torch.cuda.is_available()):
    #     checkpoint = torch.load(name)
    # else:
    #     checkpoint = torch.load(name, map_location=lambda storage, loc: storage)

    # model.load_state_dict(checkpoint['state_dict'])

    # if optimizer is not None:
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     init = checkpoint['epoch']
    #     return model, optimizer, init
    # else:
    #     return model

    def save_model(self, epoch):
        model_path = os.path.join(self.model_file, 'iter-%d' % (epoch))
        checkpoint = {
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, "../saved_models/" + name)