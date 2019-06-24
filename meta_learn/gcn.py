import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaGCN(nn.Module):
    """
    Implementation of Graph Convolutional Networks (Kipf and Welling, ICLR 2017)
    """

    def __init__(self, n_feat, n_classes, device, dropout=False, ch_list=None, multilabel=False,
                 sparse=False):
        super(MetaGCN, self).__init__()
        ch_list = ch_list or [n_feat, 256, 128]
        self.device = device
        self.multilabel = multilabel
        self.dropout = dropout
        self.gconvs = [GraphLinearLayer(
            ch_list[i], ch_list[i+1], sparse=sparse) for i in range(len(ch_list)-1)]
        self.lin = GraphLinearLayer(ch_list[-1], n_classes, sparse=sparse)
        self.layer_names = ['lin']
        # self.normalize_adj = normalize_adj
        self.eye = None
        for i, gconv in enumerate(self.gconvs):
            layer_name = 'gconv_{}'.format(i)
            self.add_module(layer_name, gconv)
            self.layer_names.append(layer_name)

    def forward(self, adj, x, param_dict=None):
        """
        Forward step of the GCN with an added functionality of using weights passed as an external
        dictionary. This is useful for inner-loop optimization in a meta-learning setting.
        :param adj: adjacency matrix
        :param x: feature matrix
        :param param_dict: a dictionary of weights to be used as model parameters
        """
        for i, gconv in enumerate(self.gconvs):
            params = None
            if param_dict is not None:
                params = param_dict['gconv_{}'.format(i)]
            x = F.relu(gconv(adj, x, params=params))
        h = self.lin(adj, x, params=param_dict['lin'] if param_dict is not None else None)
        if self.multilabel:
            h = torch.sigmoid(h)
        return h


class GraphLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, sparse=False, gpu=False):
        """
        Implementation of linear layer of GCN.
        Sparse operations are not supported yet.
        """
        super(GraphLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gpu = gpu
        self.sparse = sparse
        # TODO use dropout?
        # self.dropout = nn.Dropout(dropout)
        if sparse:
            self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        else:
            self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

    def forward(self, adj, x, params=None):
        if params is None:
            params = self.W
        # h: N x out
        if self.sparse:
            h = torch.sparse.mm(adj, x)
            h = torch.sparse.spmm(h, params)
        else:
            h = torch.mm(adj, x)
            h = torch.mm(h, params)
        return h
