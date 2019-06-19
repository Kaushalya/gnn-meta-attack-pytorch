import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from meta_learn import utils

class GNNAttack(nn.Module):
    """
    Implementation of GNN-Meta-attack
    """

    def __init__(self, model, n_nodes, train_steps=100, second_order_grad=False,
                 learning_rate=0.1, meta_learning_rate=0.005, 
                 momentum=0.9, debug=False):
        super(GNNAttack, self).__init__()
        self.model = model
        self.train_steps = train_steps
        self.second_order_grad = second_order_grad
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.momentum = momentum
        self.debug = debug
        self.adj_changes = nn.Parameter(torch.zeros(size=(n_nodes, n_nodes)))
        nn.init.xavier_normal_(self.adj_changes.data, gain=0.1)
        # self.optimizer = optim.Adam([self.adj_changes], lr=meta_learning_rate, amsgrad=False)
        self.named_weights = None

    def perturb_adj(self, adj, feature_matrix, labels, train_ids, val_ids=None):
        if self.named_weights is None:
            self.named_weights = self._get_named_param_dict(
                self.model.named_parameters())
            self.velocities = [torch.zeros_like(
                w) for w in self.named_weights.values()]
        # velocities = [torch.zeros_like(w) for w in self.named_weights.values()]
        # train the model with current adjacency matrix
        for iter in range(self.train_steps):
            self.model.zero_grad()
            preds = self.model.forward(
                adj, feature_matrix, param_dict=self.named_weights)
            loss = F.cross_entropy(preds[train_ids], labels[train_ids])
            if self.debug:
                print("epoch:{} train-loss = {}".format(iter, loss.data))
            grads = torch.autograd.grad(
                loss, self.named_weights.values(), create_graph=self.second_order_grad)
            velocities_new = [(self.momentum * v) +
                          grad for grad, v in zip(grads, self.velocities)]
            self.velocities = velocities_new
            current_params = [w - (self.learning_rate * v) for w,
                              v in zip(self.named_weights.values(), self.velocities)]
            # current_params = [w - (self.learning_rate * grad) for w, grad in
            #                   zip(self.named_weights.values(), grads)]
            self.named_weights = dict(
                zip(self.named_weights.keys(), current_params))

        # TODO calculate self-train loss
        # Sets the diagonal to zero
        adj_changes_square = self.adj_changes - torch.diag(self.adj_changes, 0)
        # Make it symmetric
        adj_changes_square = torch.clamp(
            adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)
        modified_adj = torch.clamp(adj + adj_changes_square, 0, 1)
        preds = self.model.forward(
            modified_adj, feature_matrix, param_dict=self.named_weights)
        meta_loss = F.cross_entropy(preds[train_ids], labels[train_ids])
        self.model.zero_grad()
        meta_grad = torch.autograd.grad(meta_loss, self.adj_changes)
        adj_meta_grad = meta_grad[0] * (-2*adj + 1)
        # Make sure the minimum value is 0
        adj_meta_grad -= torch.min(adj_meta_grad)
        adj_meta_grad_argmax = torch.argmax(adj_meta_grad)
        perturb_indices = np.unravel_index(
            adj_meta_grad_argmax.data, adj_meta_grad.shape)
        # flips this edge in the adjacency matrix
        adj[perturb_indices] = torch.abs(adj[perturb_indices] - 1)
        adj[perturb_indices[1], perturb_indices[0]] = adj[perturb_indices]
        # TODO filter singleton nodes
        return adj

    def forward(self, adj, feature_matrix, labels, train_ids,
                val_ids, n_perturbations=100):
        """
        performs adversarial attack.
        """
        for iter in range(n_perturbations):
            self.model.train()
            print("perturbation: {} is running".format(iter+1))
            adj = self.perturb_adj(adj, feature_matrix,
                                   labels, train_ids, val_ids)
            self.model.eval()
            preds = self.model(adj, feature_matrix)
            acc_train = utils.accuracy(preds[train_ids], labels[train_ids])
            acc_val = utils.accuracy(preds[val_ids], labels[val_ids])
            print("train-accuracy={:.2f}, val-accuracy={:.2f}".
                  format(acc_train.item(), acc_val.item()))
        return adj

    def _get_named_param_dict(self, params):
        param_dict = dict()
        for name, param in params:
            name = name.split('.')[0]
            if name in self.model.layer_names:
                param_dict[name] = param
        return param_dict
