''''DGL使用示例'''
# import dgl
# import torch
#
# #edge:(0,1),(0,3),(1,3),(1,2)
# u, v = torch.tensor([0, 0, 1, 1]), torch.tensor([1, 3, 3, 2])
# #创建图
# G = dgl.graph((u, v),num_nodes=4)
#
# print(G)
# #get nodes
# print("有向图：",G.nodes())
# #get edges
# print("有向图",G.edges())
# #有向图转为有向图
# bg = dgl.to_bidirected(G)
# print("无向图",bg.edges())
#
# #创建节点特征
# dim=64
# G.ndata['feature'] = torch.ones(G.num_nodes(), dim)
# G.edata['edgeFea'] = torch.ones(G.num_edges(), dtype=torch.int32)
# print("Add feature:",G)
#
# #利用scipy系数矩阵创建dgl图
# import scipy.sparse as sp
# spmat = sp.rand(10, 10, density=0.1) # 10%非零项
# print("scipy创建dgl：",dgl.from_scipy(spmat))

'''Dataloader使用示例'''
# import torch
# import torch.utils.data as Data
#
# Batch_Size=5
# x=torch.arange(10)
# y=torch.range(10,19,1,dtype=int)
#
# data=Data.TensorDataset(x,y)
# #将data放入DataLoader
# loader=Data.DataLoader(dataset=data,batch_size=Batch_Size,
#                        shuffle=True,num_workers=2)
#
# for epoch in range(2):
#     for batch_x,batch_y in loader:
#         print("batch_x:{},batch_y:{}".format(batch_x,batch_y))

'''GCN'''
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch
import math
import argparse
import torch.optim as optim
from torch.nn.parameter import Parameter
import dgl

#model部分，依据需求自己可以更改，建议将model部分写到一个py文件中，然后在训练的py文件中调用
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):#定义神经网络的前向传播过程，也就是一般论文中的公式实现的地方
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

#主函数，实现模型训练，一般论文中会将此部分写在train.py等文件中
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    gpu = args.gpu
    epochs=args.epochs

    #如果有gpu，用gpu，没有用cpu
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")

    #edge:(0,1),(0,3),(1,3),(1,2)
    u, v = torch.tensor([0, 0, 1, 1]), torch.tensor([1, 3, 3, 2])
    #创建图
    G = dgl.graph((u, v),num_nodes=4)
    adj = G.adjacency_matrix(transpose=True)
    features=torch.tensor([[1.,2.],[4.,5.],[7.,8.],[3.,6.]])
    labels=torch.tensor([[0,1],[1,0],[0,1],[0,1]])

    model = GCN(2, 16, 2, 0.5)  # 对应class GCN中的参数
    model.train()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=5e-4)

    for epoch in range(0,epochs):
        loss=0
        for i in range(4):
            pre = model(features, adj)
            loss_train = F.nll_loss(pre[i], labels[i])
            loss+=loss_train
            optimizer.zero_grad()
            # 反向求导  Back Propagation
            loss_train.backward()
            # 更新所有的参数
            optimizer.step()

        print(loss.data.numpy())









