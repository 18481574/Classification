import numpy as np 
import torch
import torch.nn as nn
import torch.sparse.FloatTensor as sparse

__all__ = ['GraphTV', ]

class GraphTV(nn.Module):
    def __init__(self, n=100, W=None):
        super(GraphTV, self).__init__():
            self.n = n
            self.W = W

    def forward(self, x):



    def _upd_W(self, W):
        self.W = W

    @staticmethod
    def _get_W(x, n_Neigb=15, n_sig=8):
        N, d = x.shape # N: number of samples, d: the dimension of the sample data
        if N < 2:
            raise ValueError('The number of sample must more than 1!!!')
        
        n_Neigb = min(N, n_Neigb)
        n_sig = min(n_sig, n_Neigb)

        # Calculate the distance
        X = x.numpy()
        nbrs = NearestNeighbors(n_neighbors=n_Neigb, algorithm='ball_tree').fit(X)

        dis, idx = nbrs.kneighbors(X)
        dis2 = dis**2

        M = (n_Neigb - 1) * N
        idx_i = torch.cat((torch.LongTensor(range(M)), torch.LongTensor(range(M)) ))
        idx_j = torch.cat((torch.LongTenor([idx[:,1:].repeat(n_Neigb-1), axis=0]), torch.LongTenor() ))
        

        v = torch.sqrt(v) 
        
        return sparse(torch.cat(idx_i.view(-1,1), idx_j), v)
