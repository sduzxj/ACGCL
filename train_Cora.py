import argparse, os
import math
import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from utils_mp import Subgraph, preprocess
from models import Encoder, Scorer, Pool,Encoder2,SubgraphModel
from geomloss import SamplesLoss
from function import spl_loss,balance_loss,geom_progression,linear,root_2,increase_threshold
torch.manual_seed(1024)
MAX=80
    
def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our model.')
    parser.add_argument('--dataset',help='Cora, Citeseer or Pubmed. Default=Cora', default='Cora')
    parser.add_argument('--batch_size', type=int, help='batch size', default=500)
    parser.add_argument('--subgraph_size', type=int, help='subgraph size', default=20)
    parser.add_argument('--n_order', type=int, help='order of neighbor nodes', default=10)
    parser.add_argument('--hidden_size', type=int, help='hidden size', default=1024)
    parser.add_argument('--method', help='', default='our_soft')
    
    parser.add_argument('--pre_epoch', type=int, help='pre epoch', default=0)
    parser.add_argument('--easy_epoch', type=int, help='easy epoch', default=5)
    parser.add_argument('--hard_epoch', type=int, help='hard epoch', default=50)
    parser.add_argument('--disc_func', type=str, default='w', choices=['lin', 'kl', 'w'], help='distance function for Subgraph distribution balance loss')
    return parser

    
if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        exit()
    print (args)
    
    # Loading data
    data = Planetoid(root='./dataset/' + args.dataset, name=args.dataset)
    num_classes = data.num_classes
    data = data[0]
    num_node = data.x.size(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(data.x[0])
    ppr_path = './subgraph/' + args.dataset
    subgraph = Subgraph(data.x,data.y ,data.edge_index, ppr_path, args.subgraph_size, args.n_order)
    subgraph.build()
    model = SubgraphModel(
            hidden_channels=args.hidden_size, encoder=Encoder(data.num_features, args.hidden_size),
            pool=Pool(in_channels=args.hidden_size),
            scorer=Scorer(args.hidden_size)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    global threshold
    global threshold1
    def train(epoch):
            # Model training
            model.train()
            optimizer.zero_grad()
            sample_idx = random.sample(range(data.x.size(0)), args.batch_size)
            ### Curriculum Learning Scheme
            if epoch<=args.pre_epoch:
                batch, index,batch_pos, batch_neg,y = subgraph.counterfactual_search(sample_idx,30)
                #print(y)
                z, summary = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
                loss = model.loss(z, summary)
                loss.backward()
                optimizer.step()
            else:
                if epoch<args.easy_epoch:
                 batch, index,batch_pos, batch_neg,y = subgraph.counterfactual_search(sample_idx,30)
                if epoch>=args.easy_epoch and epoch<args.hard_epoch:
                 rate=MAX*geom_progression(epoch-args.easy_epoch,args.hard_epoch-args.easy_epoch,30/MAX)
                 batch, index,batch_pos, batch_neg,y = subgraph.counterfactual_search(sample_idx,rate)
                 #print(rate)
                else :
                 batch, index,batch_pos, batch_neg,y = subgraph.counterfactual_search(sample_idx,MAX)
                threshold=1.7586436867713928
                threshold1=0.872938121395111
                for i in range(4):
                  batch_x=batch.x.cuda()
                  batch_edge_index=batch.edge_index.cuda()
                  batch_batch=batch.batch.cuda()
                  index=index.cuda()
                  batch_pos=batch_pos.cuda()
                  batch_neg=batch_neg.cuda() 
                  
                  z, summary = model(batch_x, batch_edge_index, batch_batch, index)
                  z1, summary2 = model(batch_x, batch_pos.edge_index, batch_batch, index)
                  z2, summary3 = model(batch_x, batch_neg.edge_index, batch_batch, index)
                  loss = model.loss_fn(z, summary)+torch.nn.functional.triplet_margin_loss(summary,summary2,summary3,reduction='none')
                  if epoch==1 and i==0:
                   loss_temp=loss
                   threshold=np.percentile(np.sort(loss_temp.cpu().detach().numpy()), 50)
                   threshold1=np.percentile(np.sort(loss_temp.cpu().detach().numpy()), 1/args.batch_size)*0.7
                   #print(threshold)
                   #print(threshold1)
                  v=spl_loss(loss.cuda(),args.batch_size,threshold,threshold1,args.method)
                  threshold=increase_threshold(threshold)
                  threshold1=increase_threshold(threshold1,1)
                  active_num=torch.count_nonzero(v)
                  loss=(torch.matmul(v,loss.T)+balance_loss(args.disc_func,summary,summary2)*0.2+balance_loss(args.disc_func,summary,summary3)*0.2)/active_num
                  loss.backward()
                  optimizer.step()
                  val_acc, test_acc = test(model) 
                  #print('step_acc'.format(test_acc))
                  if active_num>0.95*args.batch_size:
                        break
                  if active_num<0.1*args.batch_size:
                        break
            return loss.item()


    def get_all_node_emb(model, mask):
            # Obtain central node embs from subgraphs 
            node_list = np.arange(0, num_node, 1)[mask]
            list_size = node_list.size
            z = torch.Tensor(list_size, args.hidden_size)#.cuda() 
            group_nb = math.ceil(list_size/args.batch_size)
            for i in range(group_nb):
                maxx = min(list_size, (i + 1) * args.batch_size)
                minn = i * args.batch_size 
                batch, index,batch_pos,batch_neg,y= subgraph.counterfactual_search(node_list[minn:maxx])
                node, _ = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
                z[minn:maxx] = node
            return z


    def test(model):
            # Model testing
            model.eval()
            with torch.no_grad():
                train_z = get_all_node_emb(model, data.train_mask)
                val_z = get_all_node_emb(model, data.val_mask)
                test_z = get_all_node_emb(model, data.test_mask)

            train_y = data.y[data.train_mask]
            #print(train_y.size())
            val_y = data.y[data.val_mask]
            #print(val_y.size())
            test_y = data.y[data.test_mask]
            val_acc, test_acc = model.test(train_z, train_y, val_z, val_y, test_z, test_y)
            print('val_acc = {} test_acc = {}'.format(val_acc, test_acc))
            return val_acc, test_acc

    print('Start training !!!')
    best_acc_from_val = 0
    best_val_acc = 0
    best_ts_acc = 0
    max_val = 0
    stop_cnt = 0
    patience = 25
    for epoch in range(10000):
            loss = train(epoch)
            print('epoch = {}, loss = {}'.format(epoch, loss))
            val_acc, test_acc = test(model) 
            best_val_acc = max(best_val_acc, val_acc)
            best_ts_acc = max(best_ts_acc, test_acc)
            if val_acc >= max_val:
                max_val = val_acc
                best_acc_from_val = test_acc
                stop_cnt = 0
            else:
                stop_cnt += 1
            print('best_val_acc = {}, best_test_acc = {}'.format(best_val_acc, best_ts_acc))
            if stop_cnt >= patience:
                break
            #print('best_acc_from_val = {}'.format(best_acc_from_val))



