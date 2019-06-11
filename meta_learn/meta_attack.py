import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class GNNAttack(nn.Module):
    """
    Implementation of GNN-Meta-attack
    """
    def __init__(self, model, n_nodes, train_steps=100, second_order_grad=False,
     learning_rate=0.005, meta_learning_rate=0.005):
        super(GNNAttack, self).__init__()
        self.model = model
        self.train_steps = train_steps
        self.second_order_grad = second_order_grad
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.adj_changes = nn.Parameter(torch.zeros(size=(n_nodes, n_nodes)))
        nn.init.xavier_normal_(self.adj_changes.data, gain=1e-3)
        # self.optimizer = optim.Adam([self.adj_changes], lr=meta_learning_rate, amsgrad=False)
        self.named_weights = None
        
    def perturb_adj(self, adj, feature_matrix, train_ids, labels):
        if self.named_weights is None:
            self.named_weights = self._get_named_param_dict(self.model.named_parameters())
        # train the model with current adjacency matrix
        for iter in range(self.train_steps):
            self.model.zero_grad()
            preds = self.model.forward(adj, feature_matrix, param_dict=self.named_weights)
            loss = F.cross_entropy(preds[train_ids], labels[train_ids])
            print("epoch:{} train-loss = {}".format(iter, loss.data))
            grads = torch.autograd.grad(loss, self.named_weights.values(), retain_graph=self.second_order_grad)
            # TODO calculate velocity using momentum
            current_params = [w - self.learning_rate*grad for w, grad in zip(self.named_weights.values(), grads)]
            self.named_weights = dict(zip(self.named_weights.keys(), current_params))
        
        # TODO calculate self-train loss
        preds = self.model.forward(adj + self.adj_changes, feature_matrix, param_dict=self.named_weights)
        meta_loss = F.cross_entropy(preds[train_ids], labels[train_ids])
        # self.optimizer.zero_grad()
        # meta_loss.backward()
        self.model.zero_grad()
        meta_grad = torch.autograd.grad(meta_loss, self.adj_changes)
        self.adj_changes.data = self.adj_changes.data - self.meta_learning_rate * meta_grad[0] 
        adj_meta_grad = meta_grad[0] * (-2*adj + 1)
        # Make sure the minimum value is 0
        adj_meta_grad -= torch.min(adj_meta_grad)
        adj_meta_grad_argmax = torch.argmax(adj_meta_grad)
        perturb_indices = np.unravel_index(adj_meta_grad_argmax.data, adj_meta_grad.shape)
        # flips this edge in the adjacency matrix
        adj[perturb_indices] = torch.abs(adj[perturb_indices] - 1)
        adj[perturb_indices[1], perturb_indices[0]] = adj[perturb_indices]
        # TODO filter singleton nodes
        return adj

    def forward(self, adj, feature_matrix, train_ids, labels, n_pertubrations=100):
        """
        performs adversarial attack.
        """
        for iter in range(n_pertubrations):
            print("perturbation: {}".format(iter+1))
            adj = self.perturb_adj(adj, feature_matrix, train_ids, labels)
        return adj

    def _get_named_param_dict(self, params):
        param_dict = dict()
        for name, param in params:
            name = name.split('.')[0]
            if name in self.model.layer_names:
                param_dict[name] = param
        return param_dict
