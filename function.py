import argparse, os
import math
import torch
import random
import numpy as np
from geomloss import SamplesLoss


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