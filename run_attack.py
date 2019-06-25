from meta_learn import utils, meta_attack
import numpy as np
from meta_learn.gcn import MetaGCN
import torch
import torch.optim as optim
import torch.nn.functional as F

from argparse import ArgumentParser
from distutils.util import strtobool

def get_argparser():
    # TODO implement argument parser
    parser = ArgumentParser('GNN Meta Attack')
    parser.add_argument('--data_file', type=str, default='data/citeseer.npz')
    parser.add_argument('--use_sparse', type=strtobool, default='no',
                        help='Whether to use sparse representations')
    parser.add_argument('--normalize_adj', type=strtobool, default='yes',
                        help='Whether to normalize the adjacency matrix'+
                        ' when performing graph convolutions')
    parser.add_argument('--perturb_ratio', type=float, default=0.05)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_argparser()
    _A_obs, _X_obs, _z_obs = utils.load_npz(args.data_file)
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
    perturbations = int(args.perturb_ratio * (_A_obs.sum()//2))
    train_iters = 100
    dtype = np.float32
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Running on GPU : {}".format(device))
    
    surrogate = MetaGCN(_X_obs.shape[1], _K, device, sparse=args.use_sparse)

    if args.use_sparse:
        # Pytorch supports only sparse matrices of type COO.
        adj = utils.sparse_to_torch(_A_obs.tocoo())
        x = utils.sparse_to_torch(_X_obs.tocoo())
    else:
        adj = torch.FloatTensor(_A_obs.todense())
        x = torch.FloatTensor(_X_obs.todense())
    target = torch.LongTensor(_z_obs.astype(np.int))
    n_epoch = 20 

    if torch.cuda.is_available():
        surrogate = surrogate.to(device)
        adj = adj.to(device)
        x = x.to(device)
        target = target.to(device)

    adj_normalized = utils.preprocess_adj(adj, device=device)
    optimizer = optim.Adam(surrogate.parameters(),
                           lr=0.005,
                           weight_decay=5e-4)
    # train the model
    surrogate.train()
    for i in range(n_epoch):
        optimizer.zero_grad()
        preds = surrogate(adj_normalized, x)
        loss_train = F.cross_entropy(preds[split_train], target[split_train])
        loss_train.backward()
        optimizer.step()
        acc_train = utils.accuracy(preds[split_train], target[split_train])
        acc_val = utils.accuracy(preds[split_val], target[split_val])
        print("epoch={}, train-loss={:.3f}, train-accuracy={:.2f}, val-accuracy={:.2f}".
              format(i, loss_train.item(), acc_train.item(), acc_val.item()))

    print("Model training complete.")
    #evaluation mode
    surrogate.eval()
    preds = surrogate(adj_normalized, x)
    loss_train = F.cross_entropy(preds[split_train], target[split_train])
    acc_train = utils.accuracy(preds[split_train], target[split_train])
    acc_val = utils.accuracy(preds[split_val], target[split_val])
    print("epoch={}, train-loss={:.3f}, train-accuracy={:.2f}, \
     val-accuracy={:.2f}".format(i, loss_train.item(),
                                 acc_train.item(), acc_val.item()))
    surrogate.train()
    attacker = meta_attack.GNNAttack(surrogate, adj.shape[0], device=device,
                                     train_steps=20, learning_rate=0.1, 
                                     meta_learning_rate=0.05,
                                     second_order_grad=True, debug=True)
    
    if torch.cuda.is_available():
        attacker.cuda()

    modified_adj = attacker(adj, x, target, split_train, split_val)
