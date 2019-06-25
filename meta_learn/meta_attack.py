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

    def __init__(self, model, n_nodes, device=None, train_steps=100,
                 second_order_grad=False, learning_rate=0.1, momentum=0.9,
                 debug=False, normalize_adj=False):
        super(GNNAttack, self).__init__()
        self.model = model
        self.train_steps = train_steps
        self.second_order_grad = second_order_grad
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.debug = debug
        self.device = device
        self.normalize_adj = normalize_adj
        self.adj_changes = nn.Parameter(torch.zeros(size=(n_nodes, n_nodes)))
        nn.init.xavier_normal_(self.adj_changes.data, gain=0.1)
        self.named_weights = None

    def perturb_adj(self, adj, feature_matrix, labels, train_ids, val_ids=None):
        if self.named_weights is None:
            self.named_weights = self._get_named_param_dict(
                self.model.named_parameters())
            self.velocities = [torch.zeros_like(
                w) for w in self.named_weights.values()]
        # velocities = [torch.zeros_like(w) for w in self.named_weights.values()]
        # train the model with current adjacency matrix
        grads = None
        adj_norm = utils.preprocess_adj(adj, self.device)

        # reset gradient history
        for i, key in enumerate(self.named_weights.keys()):
            self.named_weights[key] = self.named_weights[key].detach()
            self.named_weights[key].requires_grad = True
            self.velocities[i] = self.velocities[i].detach()

        for iter in range(self.train_steps):
            self.model.zero_grad(param_dict=self.named_weights)
            preds = self.model.forward(
                adj_norm, feature_matrix, param_dict=self.named_weights)
            loss = F.cross_entropy(preds[train_ids], labels[train_ids])
            if self.debug:
                print("epoch:{} train-loss = {}".format(iter, loss.data))
            # self.model.zero_grad(param_dict=self.named_weights)
            grads = torch.autograd.grad(
                loss, self.named_weights.values(), create_graph=self.second_order_grad)
            self.velocities = [(self.momentum * v) +
                               grad for grad, v in zip(grads, self.velocities)]
            current_params = [w - (self.learning_rate * v) for w,
                              v in zip(self.named_weights.values(), self.velocities)]
            self.set_weights(current_params)

        # TODO calculate self-train loss
        # Sets the diagonal to zero
        adj_changes_square = self.adj_changes - \
            torch.diag(torch.diag(self.adj_changes, 0))
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
        # Filter self-loops
        adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
        # Filter potential singleton nodes
        singleton_mask = self.filter_singletons(adj)
        adj_meta_grad_argmax = torch.argmax(adj_meta_grad * singleton_mask)
        perturb_indices = np.unravel_index(
            adj_meta_grad_argmax.cpu().data, adj_meta_grad.shape)
        # flips this edge in the adjacency matrix
        adj[perturb_indices] = torch.abs(adj[perturb_indices] - 1)
        adj[perturb_indices[1], perturb_indices[0]] = adj[perturb_indices]

        # reset gradient history
        preds = preds.detach()
        meta_loss = meta_loss.detach()
        return adj

    def forward(self, adj, feature_matrix, labels, train_ids,
                val_ids, n_perturbations=100):
        """
        performs adversarial attack.
        """
        for iter in range(n_perturbations):
            self.model.train()
            print("perturbation: {} is running".format(iter+1))
            # self.model.zero_grad(self.named_weights)
            adj = self.perturb_adj(adj, feature_matrix,
                                   labels, train_ids, val_ids)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(adj, feature_matrix)
                acc_train = utils.accuracy(preds[train_ids], labels[train_ids])
                acc_val = utils.accuracy(preds[val_ids], labels[val_ids])
                print("train-accuracy={:.2f}, val-accuracy={:.2f}".
                      format(acc_train.item(), acc_val.item()))
        return adj

    def filter_singletons(self, adj):
        """
        Computes a mask for entries potentially leading to
        singleton nodes, i.e. one of the two nodes corresponding
        to the entry have degree 1 and there is an edge between
        the two nodes. Removing such an edge results in an isolated
        node which is not connected to the rest of the graph.
        """
        degree = torch.sum(adj, dim=0)
        deg_one = degree == 1
        row_mask = deg_one.type_as(adj).reshape((1, -1)) * adj
        column_mask = adj * deg_one.type_as(adj).reshape((-1, 1))
        mask = row_mask + column_mask
        return mask

    def set_weights(self, params):
        for key, param in zip(self.named_weights.keys(), params):
            self.named_weights[key] = param

    def _get_named_param_dict(self, params):
        param_dict = dict()
        for name, param in params:
            name = name.split('.')[0]
            if name in self.model.layer_names:
                param_dict[name] = param
        return param_dict
