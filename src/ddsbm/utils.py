import inspect
import os
from pathlib import Path
from typing import Optional, Tuple

import omegaconf
import torch
import torch_geometric.utils
import wandb
from omegaconf import OmegaConf, open_dict
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.utils import remove_self_loops, to_dense_adj, to_dense_batch


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_0":
            return self.x_0.size(0)
        if key == "edge_index_T":
            return self.x_T.size(0)
        if key == "edge_index_0_p":
            return self.x_0_p.size(0)
        if key == "edge_index_T_p":
            return self.x_T_p.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        if self.y is not None:
            self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def select_batch(self, idx):
        self.X = self.X[idx]
        self.E = self.E[idx]
        self.y = self.y[idx] if self.y is not None else None
        return self

    def extend_with_dummy(self, max_num_nodes, dummy_dim=0):
        pass


def get_k_hop_adj(adj, k):
    r"""Get degree-k adjacency matrix
    if k == 3, then it will return the adjacency matrix with degree 2 and degree 3.

    Args:
        adj (Tensor): adjacency matrix
        k (int): degree

    Returns:
        List[Tensor]: [d(2)-adj, d(3)-adj, ..., d(k)-adj]
    """
    assert k >= 2
    assert adj.size(0) == adj.size(1)
    n = adj.size(0)
    results = []
    k_hop_subgraph = adj
    k_distance_adj = adj
    for _ in range(k - 1):
        k_distance_adj = (k_distance_adj @ adj).masked_fill_(torch.eye(n).bool(), 0)
        k_distance_adj = k_distance_adj.masked_fill_(k_hop_subgraph.bool(), 0)
        k_distance_adj = k_distance_adj.masked_fill_(k_distance_adj >= 1, 1)
        k_hop_subgraph = k_hop_subgraph | (k_distance_adj == 1)
        results.append(k_distance_adj)
    return results


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs("graphs")
        os.makedirs("chains")
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs("graphs/" + args.general.name)
        os.makedirs("chains/" + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = X * norm_values[0] + norm_biases[0]
    E = E * norm_values[1] + norm_biases[1]
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch: torch.Tensor,
    max_num_nodes: Optional[int] = None,
) -> Tuple[PlaceHolder, torch.Tensor]:
    """
    Transforms a sparse graph representation into a dense graph representation.
    Additionally, it returns a node mask that indicates the presence of nodes.

    Args:
        x (Tensor): node features
        edge_index (Tensor): edge index
        edge_attr (Tensor): edge features
        batch (Tensor): batch index
        max_num_nodes (optional, int): maximum number of nodes
    Returns:
        PlaceHolder: PlaceHolder object
        node_mask (Tensor): node mask
    """

    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_num_nodes)
    # node_mask = node_mask.float()
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def to_dense_single(data, max_num_nodes=30):
    x, edge_index, edge_attr, batch = (
        data.x,
        data.edge_index,
        data.edge_attr.float(),
        data.batch,
    )
    dense_data, node_mask = to_dense(x, edge_index, edge_attr, batch, max_num_nodes)
    return dense_data, node_mask


def to_dense_pair_single(data, max_num_nodes=30, target="p"):
    """
    Transforms a PairData object into a PlaceHolder object.

    Args:
        data (PairData): PairData object
        max_num_nodes (int): maximum number of nodes
        target (str): whether to transform '0' or 'p'
    Returns:
        PlaceHolder: PlaceHolder object
    """
    if target == "0":
        x_0, edge_index_0, edge_attr_0, batch_0 = (
            data.x_0,
            data.edge_index_0,
            data.edge_attr_0.float(),
            None,
        )
        x_T, edge_index_T, edge_attr_T, batch_T = (
            data.x_T,
            data.edge_index_T,
            data.edge_attr_T.float(),
            None,
        )
        dense_data_0, node_mask_0 = to_dense(
            x_0, edge_index_0, edge_attr_0, batch_0, max_num_nodes
        )
        dense_data_T, node_mask_T = to_dense(
            x_T, edge_index_T, edge_attr_T, batch_T, max_num_nodes
        )
    elif target == "p":
        x_0, edge_index_0, edge_attr_0, batch_0 = (
            data.x_0_p,
            data.edge_index_0_p,
            data.edge_attr_0_p.float(),
            None,
        )
        x_T, edge_index_T, edge_attr_T, batch_T = (
            data.x_T_p,
            data.edge_index_T_p,
            data.edge_attr_T_p.float(),
            None,
        )
        dense_data_0, node_mask_0 = to_dense(
            x_0, edge_index_0, edge_attr_0, batch_0, max_num_nodes
        )
        dense_data_T, node_mask_T = to_dense(
            x_T, edge_index_T, edge_attr_T, batch_T, max_num_nodes
        )
    else:
        raise ValueError(f"Invalid target: {target}")

    return dense_data_0, dense_data_T, node_mask_0, node_mask_T


def to_dense_pair(data, max_num_nodes=30, target="p"):
    r"""
    Transforms a PairData object into a PlaceHolder object.
    By specifying the target, we can get end-point distribution (target='0')
    or previously-generated distribution (target='p')

    Args:
        data (PairData): PairData object
        max_num_nodes (int): maximum number of nodes
        target (str): whether to transform '0' or 'p'
    Returns:
        PlaceHolder: PlaceHolder object
    """
    if target == "0":
        x_0, edge_index_0, edge_attr_0, batch_0 = (
            data.x_0,
            data.edge_index_0,
            data.edge_attr_0.float(),
            data.get("x_0_batch", None),
        )
        x_T, edge_index_T, edge_attr_T, batch_T = (
            data.x_T,
            data.edge_index_T,
            data.edge_attr_T.float(),
            data.get("x_T_batch", None),
        )
        dense_data_0, node_mask_0 = to_dense(
            x_0, edge_index_0, edge_attr_0, batch_0, max_num_nodes
        )
        dense_data_T, node_mask_T = to_dense(
            x_T, edge_index_T, edge_attr_T, batch_T, max_num_nodes
        )
    elif target == "p":
        x_0, edge_index_0, edge_attr_0, batch_0 = (
            data.x_0_p,
            data.edge_index_0_p,
            data.edge_attr_0_p.float(),
            data.get("x_0_p_batch", None),
        )

        x_T, edge_index_T, edge_attr_T, batch_T = (
            data.x_T_p,
            data.edge_index_T_p,
            data.edge_attr_T_p.float(),
            data.get("x_T_p_batch", None),
        )
        dense_data_0, node_mask_0 = to_dense(
            x_0, edge_index_0, edge_attr_0, batch_0, max_num_nodes
        )
        dense_data_T, node_mask_T = to_dense(
            x_T, edge_index_T, edge_attr_T, batch_T, max_num_nodes
        )
    else:
        raise ValueError(f"Invalid target: {target}")

    return dense_data_0, dense_data_T, node_mask_0, node_mask_T


def _to_dense_pair(data, max_num_nodes=30):
    x_0, edge_index_0, edge_attr_0, batch_0 = (
        data.x_0,
        data.edge_index_0,
        data.edge_attr_0.float(),
        None,
    )
    x_T, edge_index_T, edge_attr_T, batch_T = (
        data.x_T,
        data.edge_index_T,
        data.edge_attr_T.float(),
        None,
    )
    dense_data_0, node_mask_0 = to_dense(
        x_0, edge_index_0, edge_attr_0, batch_0, max_num_nodes
    )
    dense_data_T, node_mask_T = to_dense(
        x_T, edge_index_T, edge_attr_T, batch_T, max_num_nodes
    )
    return dense_data_0, dense_data_T, node_mask_0, node_mask_T


def encode_no_edge(E: torch.Tensor) -> torch.Tensor:
    """
    Set no edge as the first element of the edge feature tensor.

    Args:
        E (Tensor): edge feature tensor (bs, num_nodes, num_nodes, edge_feature)
    Returns:
        Tensor: edge feature tensor with no edge encoded as the first element
    """

    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E

    # Edges should have one 1 in the last dimension
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    # Mark the first element as the no edge
    E[:, :, :, 0] = first_elt

    # Remove self-loops by setting edge features as zero vector
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )  # (bs, num_nodes, num_nodes)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


def compute_transition(M, Q):
    """M: X or E
    Compute M @ Q
    """
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(M.dtype)  # (bs, N, d) with N = n or n * n
    Q = Q.to(M.dtype)  # (bs, d, d)
    return M @ Q


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    direction = cfg.experiment.current_direction
    iteration = cfg.experiment.current_iteration
    kwargs = {
        "name": f"{direction}_{iteration}",
        "project": cfg.general.project_name,
        "config": config_dict,
        "group": cfg.general.name,  # None by default
        "settings": wandb.Settings(_disable_stats=True),
        "reinit": True,
        "mode": cfg.general.wandb,
    }
    wandb.init(**kwargs)
    wandb.save("*.txt")


def get_current_class_and_method(instance) -> str:
    """
    Returns current class and method

    Args:
        instance: instance of a class
    Returns:
        String of current class and method
    """
    current_class = instance.__class__.__name__
    current_method = inspect.currentframe().f_back.f_code.co_name
    return f"{current_class}.{current_method}"


def _collate(data_list):
    if len(data_list) == 1:
        return data_list[0], None

    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )
    return data, slices


def merge(data_path: Path, save_path: Path):
    pyg_data_dict = torch.load(data_path)

    data_list = []
    for key in pyg_data_dict:
        data_list.append(pyg_data_dict[key])

    # col = _collate(data_list)
    torch.save(_collate(data_list), save_path)
    return


def check_nll(X_0, X_T, E_0, E_T, eps=1e-18):
    bsz = X_0.size(0)
    nll_X = -torch.log(X_0 + eps) * X_T
    nll_E = -torch.log(E_0 + eps) * E_T
    nll_X = nll_X.reshape(bsz, -1).sum(dim=-1)
    nll_E = nll_E.reshape(bsz, -1).sum(dim=-1)
    nll = nll_X + nll_E
    return nll
