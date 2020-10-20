import numpy as np 
import torch
import torch.nn as nn
import torch.sparse as sparse

from sklearn.neighbors import NearestNeighbors

__all__ = ['GraphTVLoss', 'SoftMaxTV']


# Example:
# 	layer = GraphTV(n, alpha)
#   feature = feature(xs)
# 	output = model(xs) 
# 	W = Graph._get_W(feature)
# 	loss_TV = layer(output, W) 
#   loss_TV.backward()...
# 
class GraphTVLoss(nn.Module):
    def __init__(self, n=100, alpha=1., n_Neigbr=15, n_sig=8):
        super(GraphTVLoss, self).__init__()
        self.n = n
        
        self.n_Neigbr = min(n, n_Neigbr)
        self.n_sig = min(n_Neigbr, n_sig)
        self.W = None
        self.alpha = alpha 

    def forward(self, x):
        # W = self._get_W(x, self.n_Neigbr, self.n_sig)
        Wx = self.W.matmul(x)
        norm_ = Wx.norm(dim=1)

        return self.alpha *  torch.mean(norm_)


    def _upd(self, n = 100, alpha=1.0):
        self.n = n
        # self.alpha = alpha

    @staticmethod
    def _get_W(x:torch.Tensor, n_Neigbr=15, n_sig=8, target=None):
        N, d = x.shape # N: number of samples, d: the dimension of the sample data
        if N < 2:
            raise ValueError('The number of sample must more than 1!!!')
        
        n_Neigb = min(N, n_Neigbr)
        n_sig = min(n_sig, n_Neigbr)

        # Calculate the distance
        X = x.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=n_Neigbr, algorithm='ball_tree').fit(X)

        dis, idx = nbrs.kneighbors(X)
        dis2 = np.exp(- dis**2 / dis[:, n_sig-1:n_sig] / 2 )

        if target is None:
            M = (n_Neigbr - 1) * N
            idx_i = torch.cat( (torch.LongTensor(range(M)), torch.LongTensor(range(M)), ) )
            idx_j = torch.cat( (torch.LongTensor(idx[:, 0].repeat(n_Neigbr-1)), torch.LongTensor(idx[:,1:].repeat(1)), ) )
        
            v = torch.cat( (torch.FloatTensor(dis2[:, 1:] ).view( -1), - torch.FloatTensor(dis2[:, 1:]).view(-1), ) )
            # print(idx_i.shape, idx_j.shape, v.shape)
        else:
            idx_i = []
            idx_j = []
            v = []

            tot = 0
            for i in range(N):
                for k in range(1, n_Neigbr):
                    j = idx[i,k]

                    if target[i] == target[j]:
                        idx_i += [tot, tot]
                        idx_j += [i, j]
                        v += [dis2[i,k], -dis2[i,k]]

                        tot += 1

            idx_i = torch.LongTensor(idx_i)
            idx_j = torch.LongTensor(idx_j)
            v = torch.FloatTensor(v)
            M = tot

        return sparse.FloatTensor(torch.cat( (idx_i.view(1, -1), idx_j.view(1, -1)) ), v, torch.Size([M, N]))


class SoftMaxTV(nn.Module):
    def __init__(self, alpha=1., tau=.1, iter=10, W=None):
        super(SoftMaxTV, self).__init__()
        self.W = W
        self.alpha = alpha
        self.tau = tau*alpha
        self.iter = iter

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        grad = self.W.matmul(self.softmax(x))
        eta = torch.zeros_like(grad)

        for i in range(self.iter):
            eta = nn.functional.normalize(eta + self.tau * grad)
            x =  x - self.alpha * self.W.transpose(0, 1).matmul(grad)
            grad = self.W.matmul(self.softmax(x))

        return x  # return logit instead of SoftMax