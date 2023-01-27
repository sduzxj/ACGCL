import random
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling
EPS = 1e-15


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv = GCNConv(in_channels, self.hidden_channels) 
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels) 
        self.prelu = nn.PReLU(self.hidden_channels)

    def forward(self, x, edge_index):
        x1 = self.conv(x, edge_index)
        #x1 = F.dropout(x1,p=0.1)
        x1 = self.prelu(x1)
        return x1 
  
class Encoder2(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder2, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv = GCNConv(in_channels, self.hidden_channels) 
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels) 
        self.prelu = nn.PReLU(self.hidden_channels)

    def forward(self, x, edge_index):
        x1 = self.conv(x, edge_index)
        #x1 = F.dropout(x1,p=0.1)
        x1 = self.prelu(x1)
        #x1 = F.dropout(x1,p=0.2)
        ###
        x1 = self.conv2(x1, edge_index)
        #x1 = F.dropout(x1,p=0.1)
        x1 = self.prelu(x1)
        ###
        return x1         
class Encoder3(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder3, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv = GCNConv(in_channels, self.hidden_channels) 
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels) 
        self.prelu = nn.PReLU(self.hidden_channels)

    def forward(self, x, edge_index):
        x1 = self.conv(x, edge_index)
        x1 = F.dropout(x1,p=0.1)
        x1 = self.prelu(x1)
        #x1 = F.dropout(x1,p=0.2)
        ###
        x1 = self.conv2(x1, edge_index)
        x1 = F.dropout(x1,p=0.1)
        x1 = self.prelu(x1)
        ###
        return x1   
class Pool(nn.Module):
    def __init__(self, in_channels, ratio=1.0):
        super(Pool, self).__init__()
        self.sag_pool = SAGPooling(in_channels, ratio)
        self.lin1 = torch.nn.Linear(in_channels * 2, in_channels)
        
    def forward(self, x, edge, batch, type='mean_pool'):
        if type == 'mean_pool':
            return global_mean_pool(x, batch)
        elif type == 'max_pool':
            return global_max_pool(x, batch)
        elif type == 'sum_pool':
            return global_add_pool(x, batch)
        elif type == 'sag_pool':
            x1, _, _, batch, _, _ = self.sag_pool(x, edge, batch=batch)
            return global_mean_pool(x1, batch)
            
class Encoder_gcl(nn.Module):
    def __init__(self, in_channels, out_channels, k, skip=True):
        super(Encoder_gcl, self).__init__()
        self.base_model = GCNConv
        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            # self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            self.conv = [self.base_model(in_channels, 2 * out_channels)]
            for _ in range(1, k - 1):
                self.conv.append(self.base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(self.base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = nn.PReLU(out_channels)
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [self.base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(self.base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]            
        
class GraphSAGE_GCN(torch.nn.Module):
    def __init__(self, input_dim, layer_num=3, hidden=512):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.layer = layer_num
        self.acts = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for i in range(self.layer):
            if i == 0:
                self.convs.append(SAGEConv(input_dim, hidden, root_weight=True))
            else:
                self.convs.append(SAGEConv(hidden, hidden, root_weight=True))
            # self.acts.append(torch.nn.PReLU(hidden))
            self.acts.append(torch.nn.ELU())
            self.norms.append(torch.nn.BatchNorm1d(hidden))
            
    def forward(self, x,edge_index):
        for i in range(self.layer):
            x = self.acts[i](self.norms[i](self.convs[i](x, edge_index)))
        return x        


class Scorer(nn.Module):
    def __init__(self, hidden_size):
        super(Scorer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

    def forward(self, input1, input2):
        output = torch.sigmoid(torch.sum(input1 * torch.matmul(input2, self.weight), dim = -1))
        return output


class SubgraphModel(torch.nn.Module):

    def __init__(self, hidden_channels, encoder, pool, scorer):
        super(SubgraphModel, self).__init__()
        self.encoder = encoder
        self.hidden_channels = hidden_channels
        self.pool = pool
        self.scorer = scorer
        self.marginloss = nn.MarginRankingLoss(0.5)
        self.marginloss_fn = nn.MarginRankingLoss(0.5,reduction ='none')
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
        #self.Linear(hidden_channels,1)
        
    def reset_parameters(self):
        reset(self.scorer)
        reset(self.encoder)
        reset(self.pool)
        
    def forward(self, x, edge_index, batch=None, index=None):
        r""" Return node and subgraph representations of each node before and after being shuffled """
        hidden = self.encoder(x, edge_index)
        if index is None:
            return hidden
        
        z = hidden[index]
        summary = self.pool(hidden, edge_index, batch)
        return z, summary
    
    def loss(self, hidden1, summary1):
        r"""Computes the margin objective."""

        shuf_index = torch.randperm(summary1.size(0))

        hidden2 = hidden1[shuf_index]
        summary2 = summary1[shuf_index]
        
        logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim = -1))
        logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim = -1))
        logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim = -1))
        logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim = -1))
        
        TotalLoss = 0.0
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        TotalLoss += self.marginloss(logits_aa, logits_ba, ones)
        TotalLoss += self.marginloss(logits_bb, logits_ab, ones)
        
        return TotalLoss
    def loss_fn(self, hidden1, summary1):
        r"""Computes the margin objective."""

        shuf_index = torch.randperm(summary1.size(0))

        hidden2 = hidden1[shuf_index]
        summary2 = summary1[shuf_index]
        
        logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim = -1))
        logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim = -1))
        logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim = -1))
        logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim = -1))
        TotalLoss = 0.0
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        TotalLoss += self.marginloss_fn(logits_aa, logits_ba, ones)
        TotalLoss += self.marginloss_fn(logits_bb, logits_ab, ones)
        
        return TotalLoss

    def test(self, train_z, train_y, val_z, val_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream task."""        
        clf = LogisticRegression(solver=solver, multi_class=multi_class,max_iter=1000,*args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        val_acc = clf.score(val_z.detach().cpu().numpy(), val_y.detach().cpu().numpy())
        test_acc = clf.score(test_z.detach().cpu().numpy(), test_y.detach().cpu().numpy())
        return val_acc, test_acc
    '''
    def test_f1(self, train_z, train_y, val_z, val_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream task."""        
        clf = LogisticRegression(solver=solver, multi_class=multi_class,max_iter=2000,*args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        pro_t_y=clf.predict(test_z.detach().cpu().numpy())
        pro_val_y=clf.predict(val_z.detach().cpu().numpy())
        val_acc = f1_score(pro_val_y, val_y.detach().cpu().numpy(),average='micro')
        test_acc = f1_score(pro_t_y, test_y.detach().cpu().numpy(),average='micro')
        return val_acc, test_acc'''
