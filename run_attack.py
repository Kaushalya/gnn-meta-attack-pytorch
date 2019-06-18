from meta_learn import utils, meta_attack
import numpy as np
from meta_learn.gcn import MetaGCN
import torch
import torch.optim as optim
import torch.nn.functional as F

if __name__ == "__main__":
    _A_obs, _X_obs, _z_obs = utils.load_npz('data/citeseer.npz')
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    lcc = utils.largest_connected_components(_A_obs)

    _A_obs = _A_obs[lcc][:,lcc]
    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32")
    _A_obs.eliminate_zeros()
    _X_obs = _X_obs.astype("float32")

    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
    assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    _X_obs = _X_obs[lcc]
    _z_obs = _z_obs[lcc]
    _N = _A_obs.shape[0]
    _K = _z_obs.max()+1
    _Z_obs = np.eye(_K)[_z_obs]
    _An = utils.preprocess_adj(_A_obs)
    # sizes = [16, _K]
    degrees = _A_obs.sum(0).A1

    seed = 15
    unlabeled_share = 0.8
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share
    np.random.seed(seed)

    split_train, split_val, split_unlabeled = utils.train_val_test_split_tabular(np.arange(_N),
                                                                        train_size=train_share,
                                                                        val_size=val_share,
                                                                        test_size=unlabeled_share,
                                                                        stratify=_z_obs)
    split_unlabeled = np.union1d(split_val, split_unlabeled)
    
    hidden_sizes = [16]
    use_sparse = False
    share_perturbations = 0.05
    perturbations = int(share_perturbations * (_A_obs.sum()//2))
    train_iters = 100
    dtype = np.float32
    gpu_id = -1
    surrogate = MetaGCN(_X_obs.shape[1], _K, sparse=use_sparse)
    optimizer = optim.Adam(surrogate.parameters(), 
                       lr=0.005, 
                       weight_decay=5e-4)
    
    if use_sparse:
        # Pytorch supports only sparse matrices of type COO.
        adj = utils.sparse_to_torch(_A_obs.tocoo())
        x = utils.sparse_to_torch(_X_obs.tocoo())
    else:
        adj = torch.FloatTensor(_A_obs.todense())
        x = torch.FloatTensor(_X_obs.todense())
    target = torch.LongTensor(_z_obs.astype(np.int))
    n_epoch = 50 

    # train the model
    surrogate.train()
    for i in range(n_epoch):
        optimizer.zero_grad()
        preds = surrogate(adj, x)
        loss_train = F.cross_entropy(preds[split_train], target[split_train])
        loss_train.backward()
        optimizer.step()
        acc_train = utils.accuracy(preds[split_train], target[split_train])
        acc_val = utils.accuracy(preds[split_val], target[split_val])
        print("epoch={}, train-loss={:.3f}, train-accuracy={:.2f}, val-accuracy={:.2f}".format(i, loss_train.item(),
         acc_train.item(), acc_val.item()))
    
    print("Model training complete.")
    #evaluation mode
    surrogate.eval()
    preds = surrogate(adj, x)
    loss_train = F.cross_entropy(preds[split_train], target[split_train])
    acc_train = utils.accuracy(preds[split_train], target[split_train])
    acc_val = utils.accuracy(preds[split_val], target[split_val])
    print("epoch={}, train-loss={:.3f}, train-accuracy={:.2f}, val-accuracy={:.2f}".format(i, loss_train.item(),
                       
                                                                                           acc_train.item(), acc_val.item()))
    surrogate.train()
    attacker = meta_attack.GNNAttack(surrogate, adj.shape[0], train_steps=50, learning_rate=0.01,
                                     meta_learning_rate=0.05, second_order_grad=True)
    attacker(adj, x, split_train, target)

