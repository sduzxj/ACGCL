import argparse, os
import math
import torch
import random
import numpy as np
from geomloss import SamplesLoss
from typing import Optional
from torch.optim import Adam
import torch.nn as nn

def linear(x, t, c0):
	return (x* ((1-c0)/t)) + c0

def root_2(x, t, c0):
	return ((x* ((1-(c0**2.0))/t)) + (c0**2.0))**(1./2)

def root_5(x, t, c0):
	return ((x* ((1-(c0**5.0))/t)) + (c0**5.0))**(1./5)

def root_10(x, t, c0):
	return ((x* ((1-(c0**10.0))/t)) + (c0**10.0))**(1./10)

def root_20(x, t, c0):
	return ((x* ((1-(c0**20.0))/t)) + (c0**20.0))**(1./20)

def root_50(x, t, c0):
	return ((x* ((1-(c0**50.0))/t)) + (c0**50.0))**(1./50)

def geom_progression(x, t, c0):
	return 2.0**((x* ((math.log(1,2.0)-math.log(c0,2.0))/t)) +math.log(c0,2.0))

def quadratic(x, t, c0):
	return (x* ((1-c0**1.54)/t))**2 + c0

def cubic(x, t, c0):
	return (x* ((1-c0**1.87)/t))**3 + c0

def increase_threshold(threshold,growing_factor=1.1):
        threshold *= growing_factor
        return threshold
'''
MAX*linear(epoch-args.easy_epoch,args.hard_epoch-args.easy_epoch,10/MAX)
MAX*root_2(epoch-args.easy_epoch,args.hard_epoch-args.easy_epoch,10/MAX)
MAX*geom_progression(epoch-args.easy_epoch,args.hard_epoch-args.easy_epoch,10/MAX)
MAX*quadratic(epoch-args.easy_epoch,args.hard_epoch-args.easy_epoch,10/MAX)
MAX*cubic(epoch-args.easy_epoch,args.hard_epoch-args.easy_epoch,0.1)
'''
#### Subgraph distribution balance loss 
def balance_loss(disc_func, nodepairs_f, nodepairs_cf):
    X_f = nodepairs_f
    X_cf = nodepairs_cf
    if disc_func == 'lin':
        mean_f = X_f.mean(0)
        mean_cf = X_cf.mean(0)
        loss_disc = torch.sqrt(F.mse_loss(mean_f, mean_cf) + 1e-6)
    elif disc_func == 'kl':
        # kl divergence
        pass
    elif disc_func == 'w':
        # Wasserstein distance
        dist = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        loss_disc = dist(X_cf, X_f)
    else:
        raise Exception('unsupported distance function for discrepancy loss')
    return loss_disc
    

def spl_loss(super_loss,batch_size,threshold=0.5,threshold1=0,method='hard'):
    
        if method=='Mix':
         ones=torch.ones(batch_size).cuda()
         threshold_tensor=threshold*ones
         threshold1_tensor=threshold1*ones
         v = (super_loss <threshold1_tensor).int()
         lamba=(threshold*threshold1)/(threshold-threshold1)
        
         v=lamba*(torch.div(ones,super_loss)-(1/threshold)*ones)*((super_loss>threshold1_tensor)*(super_loss<threshold_tensor)).int()+v
         #print(v)
        
        if method=='Linear':
            
         ones=torch.ones(batch_size).cuda()
         threshold_tensor=threshold*ones
         threshold1_tensor=threshold1*ones
         v = (super_loss <threshold_tensor).int()
         v=(ones-super_loss/threshold)*(super_loss<threshold_tensor).int()
        
        if method=='hard':
         threshold=threshold*torch.ones(batch_size).cuda()
         threshold1=threshold1*torch.ones(batch_size).cuda()
         v = ((super_loss <threshold) * (super_loss>threshold1))
            
        if method=='our_soft':
         threshold_tensor=threshold*torch.ones(batch_size).cuda()
         threshold1_tensor=threshold1*torch.ones(batch_size).cuda()
         v = ((super_loss <threshold_tensor) * (super_loss>threshold1_tensor)).int()
         v=(super_loss/threshold1)*(super_loss<threshold1_tensor).int()+v
        return v.float()
    

def get_idx_split(dataset, split):
    if split[:4] == 'rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = dataset[0].x.size(0)
        train_size = int(num_nodes * train_ratio)
        indices = torch.randperm(num_nodes)
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }
    elif split == 'ogb':
        return dataset.get_idx_split()
    elif split.startswith('wikics'):
        split_idx = int(split.split(':')[1])
        return {
            'train': dataset[0].train_mask[:, split_idx],
            'test': dataset[0].test_mask,
            'val': dataset[0].val_mask[:, split_idx]
        }
    else:
        raise RuntimeError(f'Unknown split type {split}')
        
def log_regression(z,
                   dataset,
                   evaluator,
                   num_epochs: int = 5000,
                   test_device: Optional[str] = None,
                   split: str = 'rand:0.1',
                   verbose: bool = False,
                   preload_split=None):
    test_device = z.device if test_device is None else test_device
    z = z.detach().to(test_device)
    num_hidden = z.size(1)
    y = dataset[0].y.view(-1).to(test_device)
    num_classes = dataset[0].y.max().item() + 1
    classifier = LogReg(num_hidden, num_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    ###split = dataset.get_idx_split()
    split=get_idx_split(dataset,split)
    split = {k: v.to(test_device) for k, v in split.items()}
    #print(split['train'].sum())
    #print(split['test'].sum())
    #print(split['valid'].sum())
    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()

    best_test_acc = 0
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()

        output = classifier(z[split['train']])
        loss = nll_loss(f(output), y[split['train']])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            if 'val' in split:
                # val split is available
                test_acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                val_acc = evaluator.eval({
                    'y_true': y[split['val']].view(-1, 1),
                    'y_pred': classifier(z[split['val']]).argmax(-1).view(-1, 1)
                })['acc']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
            else:
                acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                if best_test_acc < acc:
                    best_test_acc = acc
                    best_epoch = epoch
            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_acc}')

    return {'acc': best_test_acc}

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
    
def test_arxiv(final=False):
    model.eval()
    z = model(data.x, data.edge_index)
    nclass = num_classes
    evaluator = MulticlassEvaluator(n_clusters=nclass, random_state=0, n_jobs=8)
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']
    return acc

class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        return (correct / total).item()

    def eval(self, res):
        return {'acc': self._eval(**res)}
    
