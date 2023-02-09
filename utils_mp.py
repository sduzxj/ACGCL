import os 
import torch
import numpy as np
from cytoolz import curry
import multiprocessing as mp
from scipy import sparse as sp
from sklearn.preprocessing import normalize, StandardScaler
from torch_geometric.data import Data, Batch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from multiprocessing import Pool
from itertools import combinations


def standardize(feat, mask):
    scaler = StandardScaler()
    scaler.fit(feat[mask])
    new_feat = torch.FloatTensor(scaler.transform(feat))
    return new_feats
    
    
def preprocess(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.tensor(features)


def computecf_graph(self, x,y,edge,thresh=30,subgraph_node_num=20,alpha=1):
    #Calculate the counterfactual graph
            node_embs=x
            simi_mat = cdist(node_embs, node_embs, 'euclidean')#minkowski
            adjn=x.shape[0]
            edgen=edge[0].shape[0]
            adj=np.zeros([adjn,adjn])
            for i in range(edgen):
                  adj[edge[0][i]][edge[1][i]]=1
                  adj[edge[1][i]][edge[0][i]]=1
            thresh = np.percentile(np.sort(simi_mat), thresh)
            np.fill_diagonal(simi_mat, np.max(simi_mat)+1)
            node_nns = np.argsort(simi_mat, axis=1)
            adj_cf = np.zeros([adjn,adjn])
            node_pairs = list(combinations(range(adjn), 2))
            edges_cf_t0a = []
            edges_cf_t0b = []
            edges_cf_t1a = []
            edges_cf_t1b = []
            c = 0
            type1=0
            type2=0
            type3=0
            for a, b in node_pairs:
                    nns_a = node_nns[a]
                    nns_b = node_nns[b]
                    i, j = 0, 0
                    while i < len(nns_a)-1 and j < len(nns_b)-1:
                        if simi_mat[a, nns_a[i]] + simi_mat[b, nns_b[j]] > alpha*thresh:
                            #T_cf[a, b] = T_f[a, b]
                            adj_cf[a, b] = adj[a, b]
                            if adj_cf[a, b]==1:
                                edges_cf_t0a.append(a)
                                edges_cf_t0b.append(b)
                                edges_cf_t1a.append(a)
                                edges_cf_t1b.append(b)
                            break
                        if [y[nns_a[i]], y[nns_b[j]]] != [y[a], y[b]]:
                            adj_cf[a, b] = adj[nns_a[i], nns_b[j]]
                            if adj_cf[a, b]==1:
                              edges_cf_t0a.append(a)
                              edges_cf_t0b.append(b)
                            break
                        if [y[nns_a[i]], y[nns_b[j]]] == [y[a], y[b]]:
                            adj_cf[a, b] = adj[a, b]
                            if adj_cf[a, b]==1:
                              edges_cf_t1a.append(a)
                              edges_cf_t1b.append(b)
                        if simi_mat[a, nns_a[i+1]] < simi_mat[b, nns_b[j+1]]:
                            i += 1
                        else:
                            j += 1
                    c += 1
            ecf0=torch.IntTensor([edges_cf_t0a,edges_cf_t0b]).type(torch.long)
            ecf1=torch.IntTensor([edges_cf_t1a,edges_cf_t1b]).type(torch.long)
            return ecf1,ecf0

class PPR:
    #Node-wise personalized pagerank
    def __init__(self, adj_mat, maxsize=200, n_order=2, alpha=0.85):
        self.n_order = n_order
        self.maxsize = maxsize
        self.adj_mat = adj_mat
        self.P = normalize(adj_mat, norm='l1', axis=0)
        self.d = np.array(adj_mat.sum(1)).squeeze()
        
    def search(self, seed, alpha=0.85):
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1])
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)
        
        idx = scores.argsort()[::-1][:self.maxsize]
        neighbor = np.array(x.indices[idx])
        
        seed_idx = np.where(neighbor == seed)[0]
        if seed_idx.size == 0:
            neighbor = np.append(np.array([seed]), neighbor)
        else :
            seed_idx = seed_idx[0]
            neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]
            
        assert np.where(neighbor == seed)[0].size == 1
        assert np.where(neighbor == seed)[0][0] == 0
        
        return neighbor
    
    @curry
    def process(self, path, seed):
        ppr_path = os.path.join(path, 'ppr{}'.format(seed))
        if not os.path.isfile(ppr_path) or os.stat(ppr_path).st_size == 0:
            print ('Processing node {}.'.format(seed))
            neighbor = self.search(seed)
            torch.save(neighbor, ppr_path)
        else :
            print ('File of node {} exists.'.format(seed))
    
    def search_all(self, node_num, path):
        neighbor  = {}
        if os.path.isfile(path+'_neighbor') and os.stat(path+'_neighbor').st_size != 0:
            print ("Exists neighbor file")
            neighbor = torch.load(path+'_neighbor')
        else :
            print ("Extracting subgraphs")
            os.system('mkdir {}'.format(path))
            with mp.Pool() as pool:
                list(pool.imap_unordered(self.process(path), list(range(node_num)), chunksize=1000))
                
            print ("Finish Extracting")
            for i in range(node_num):
                neighbor[i] = torch.load(os.path.join(path, 'ppr{}'.format(i)))
            torch.save(neighbor, path+'_neighbor')
            os.system('rm -r {}'.format(path))
            print ("Finish Writing")
        return neighbor

    
class Subgraph:
    #Class for subgraph extraction
    
    def __init__(self, x,y, edge_index, path, maxsize=50, n_order=10):
        self.x = x
        self.y = y
        self.path = path
        self.edge_index = np.array(edge_index)
        self.edge_num = edge_index[0].size(0)
        self.node_num = x.size(0)
        self.maxsize = maxsize
        
        self.sp_adj = sp.csc_matrix((np.ones(self.edge_num), (edge_index[0], edge_index[1])), 
                                    shape=[self.node_num, self.node_num])
        self.ppr = PPR(self.sp_adj, n_order=n_order)
        
        self.neighbor = {}
        self.adj_list = {}
        self.subgraph = {}
        self.subgraph_pos={}
        self.subgraph_neg={}
        
    def process_adj_list(self):
        for i in range(self.node_num):
            self.adj_list[i] = set()
        for i in range(self.edge_num):
            u, v = self.edge_index[0][i], self.edge_index[1][i]
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)
            
    def adjust_edge(self, idx):
        #Generate edges for subgraphs
        dic = {}
        for i in range(len(idx)):
            dic[idx[i]] = i
            
        new_index = [[], []]
        nodes = set(idx)
        for i in idx:
            edge = list(self.adj_list[i] & nodes)
            edge = [dic[_] for _ in edge]
            #edge = [_ for _ in edge if _ > i]
            new_index[0] += len(edge) * [dic[i]]
            new_index[1] += edge
        return torch.LongTensor(new_index)

    def adjust_x(self, idx):
        #Generate node features for subgraphs
        return self.x[idx]            
    def adjust_y(self, idx):
        #Generate node features for subgraphs
        return self.y[idx] 
    
    def build(self):
        #Extract subgraphs for all nodes
        if os.path.isfile(self.path+'_subgraph') and os.stat(self.path+'_subgraph').st_size != 0:
            print ("Exists subgraph file")
            self.subgraph = torch.load(self.path+'_subgraph')
            #self.subgraph_pos = torch.load(self.path+'_subgraph_pos')
            #self.subgraph_neg = torch.load(self.path+'_subgraph_neg')
            return 

        self.neighbor = self.ppr.search_all(self.node_num, self.path)
        self.process_adj_list()
        for i in range(self.node_num):
            nodes = self.neighbor[i][:self.maxsize]
            x = self.adjust_x(nodes)
            edge = self.adjust_edge(nodes)
            y = self.adjust_y(nodes)
            self.subgraph[i] = Data(x, edge, y=y)
            #subcf_pos_edge,subcf_neg_edge=computecf_graph(self,x=self.subgraph[i].x,y=y,edge=edge)
            #self.subgraph_pos[i]=Data(x,subcf_pos_edge)
            #self.subgraph_neg[i]=Data(x,subcf_neg_edge)
        torch.save(self.subgraph, self.path+'_subgraph')
        #torch.save(self.subgraph_pos, self.path+'_subgraph_pos')
        #torch.save(self.subgraph_neg, self.path+'_subgraph_neg')
        
    def counterfactual_search(self, node_list,thresh=30):
        #Extract subgraphs and the corresponding counterfactual graphs for nodes in the list
        batch = []
        index = []
        
        batch_pos = []
        batch_neg=[]
        y=[]
        
        size = 0
        for node in node_list:
            #x = self.adjust_x(nodes)
            #edge = self.adjust_edge(node)
            #y = self.adjust_y(nodes)
            subcf_pos_edge,subcf_neg_edge=computecf_graph(self,x=self.subgraph[node].x,y=self.subgraph[node].y,edge=self.subgraph[node].edge_index,thresh=
                                                         thresh,subgraph_node_num=self.maxsize)
            batch.append(self.subgraph[node])
            batch_pos.append(Data(self.subgraph[node].x,subcf_pos_edge))
            batch_neg.append(Data(self.subgraph[node].x,subcf_neg_edge))
            index.append(size)
            y.append(self.subgraph[node].y)
            size += self.subgraph[node].x.size(0)
        index = torch.tensor(index)
        batch = Batch().from_data_list(batch)
        batch_pos = Batch().from_data_list(batch_pos)
        batch_neg = Batch().from_data_list(batch_neg)

        return batch, index,batch_pos,batch_neg,y