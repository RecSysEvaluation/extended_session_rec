import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import os


from torch.nn import Module, Parameter
import torch.nn.functional as F
from algorithms.CM_HGCN.aggregator import LocalAggregator, GlobalAggregator,GNN,LocalAggregator_mix

class CombineGraph(Module):
    def __init__(self, num_nodes, n_category, category, lr, batch_size, l2, dim):
        super(CombineGraph, self).__init__()
        
        self.batch_size = batch_size
 #      self.num_node = num_node
        self.num_total = num_nodes
        self.dim = dim # hidden size
        self.dropout_local = 0
        self.dropout_global = 0
        self.hop = 1
        self.sample_num = 12
        self.alpha = 0.2
        self.n_category = n_category
        self.category = category
        self.lr = lr
        self.l2 = l2
        self.lr_dc = 0.1
        self.lr_dc_step = 3
        
        
        # Aggregator
        self.local_agg_1 = LocalAggregator(self.dim, self.alpha, dropout=0.0)
       # self.local_agg_2 = LocalAggregator(50, self.opt.alpha, dropout=0.0)
        self.gnn = GNN(self.dim)
        
        self.local_agg_mix_1 = LocalAggregator(self.dim, self.alpha, dropout=0.0)
   #    self.local_agg_mix_2 = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
   #     self.local_agg_mix_3 = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)

        # Item representation & Position representation
        self.embedding = nn.Embedding(self.num_total, self.dim)
        self.pos = nn.Embedding(200, self.dim)
        

        # Parameters_1
        self.w_1 = nn.Parameter(torch.Tensor(3 * self.dim, 2*self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(2*self.dim, 1))
        self.glu1 = nn.Linear(2*self.dim, 2*self.dim)
        self.glu2 = nn.Linear(2*self.dim, 2*self.dim, bias=False)
   #     self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)
        

        
 
       # self.aaa = Parameter(torch.Tensor(1))
        self.bbb = Parameter(torch.Tensor(1))
        self.ccc = Parameter(torch.Tensor(1))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay= self.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size= self.lr_dc_step, gamma= self.lr_dc)
        self.reset_parameters()
    
        item = []
        for x in range(1, self.num_total + 1 - n_category):
            item += [category[x]]
        item = np.asarray(item)  
        self.item =  trans_to_cuda(torch.Tensor(item).long())

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden1, hidden2, hidden1_mix, hidden2_mix, mask):
        hidden1 = hidden1 + hidden1_mix * self.bbb
        hidden2 = hidden2 + hidden2_mix * self.ccc
        hidden = torch.cat([hidden1, hidden2],-1)
        
        mask = mask.float().unsqueeze(-1)
        batch_size = hidden1.shape[0]
        len = hidden1.shape[1]
        
        pos_emb = self.pos.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        # Error is here for eCommerce dataset.....
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)

        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        b = self.embedding.weight[1:self.num_total-self.n_category+1]  # n_nodes x latent_size
        item_category = self.embedding(self.item)     #n*d
        
        t = torch.cat([b,item_category],-1)
        scores = torch.matmul(select, t.transpose(1, 0))

        return scores

    def forward(self, inputs, adj, mask_item, item, items_ID, adj_ID, total_items, total_adj):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        
        hidden1 = self.embedding(inputs)
        hidden2 = self.embedding(items_ID)
        hidden_mix = self.embedding(total_items)
        
        # local
        hidden1 = self.local_agg_1(hidden1, adj, mask_item)
  #      hidden2 = self.local_agg_2(hidden2, adj_ID, mask_item)
        hidden2 = self.gnn(adj_ID,hidden2)
        
        hidden_mix = self.local_agg_mix_1(hidden_mix, total_adj, mask_item)
    #    hidden_mix = self.local_agg_mix_2(hidden_mix, total_adj, mask_item)
    #    hidden_mix = self.local_agg_mix_3(hidden_mix, total_adj, mask_item)

        # combine
        hidden1 = F.dropout(hidden1, self.dropout_local, training=self.training)
        hidden2 = F.dropout(hidden2, self.dropout_local, training=self.training)
        hidden_mix = F.dropout(hidden_mix, self.dropout_local, training=self.training)

        return hidden1, hidden2, hidden_mix
    
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
    
    
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable    
    




