import warnings

import pygmtools as pygm
import torch  # pytorch backend
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import MessagePassing

from ddsbm import utils
from ddsbm.graph_match.utils import construct_K

pygm.set_backend("pytorch")  # set default backend for pygmtools


class MPMModule(MessagePassing):
    def __init__(
        self,
        pooling_type="max",
        max_iter=1000,
        # tol=1e-6,  # HACK: REMOVED (at 241113)
        noise_coeff=1e-6,
        dtype=torch.float32,
    ):
        super(MPMModule, self).__init__(aggr="sum", flow="target_to_source")
        self.pooling_type = pooling_type
        self.max_iter = max_iter
        # self.tol = tol  # HACK: REMOVED (at 241113)
        self.noise_coeff = noise_coeff
        self.dtype = dtype

    def forward(self, x, edge_index, edge_feat):
        return self.propagate(edge_index, x=x, edge_feat=edge_feat)

    def message(self, x_i, x_j, edge_feat):
        n = x_j.shape[-1]
        edge_feat = edge_feat.reshape(-1, n, n)
        if self.pooling_type == "max":
            return (x_j.unsqueeze(1) * edge_feat).max(dim=-1, keepdim=False).values

        elif self.pooling_type == "sum":
            return (x_j.unsqueeze(1) * edge_feat).sum(dim=-1, keepdim=False)

        else:
            raise ValueError(
                "Invalid pooling type, pooling_type should be one of ['max', 'sum']"
            )

    def update(self, aggr_out):
        return aggr_out

    def solve(
        self, K, edge_index, x=None, bsz=None, max_iter=1000, tol=1e-8, noise_coeff=1e-6
    ):
        r"""
        K : cost matrix with shape (num_edges, max_node * batch_size, max_node)
        edge_index : edge index with shape (2, num_edges)
        x : initial matching matrix with shape (max_node * batch_size, max_node)
        """
        K = K.to(self.dtype)
        # assert K.dtype == torch.float64, f"K.dtype : {K.dtype}"
        edge_index = edge_index.to(K.device)

        # init x
        if x is None:
            n = int(K.size(1) ** 0.5)
            if bsz is None:
                raise ValueError("bsz must be provided when x is None")
            x = torch.ones(bsz * n, n).to(K.device).to(K.dtype)

        else:
            n = x.size(1)
            bsz = x.size(0) // n
            assert x.size(0) % n == 0
            x = x.to(K.device).to(K.dtype)

        # noise_coeff = 1e-7
        if noise_coeff > 0:
            K = K + noise_coeff * torch.rand_like(K)

        x = x.reshape(bsz, n**2)
        norm = x.norm(p=2, dim=-1, keepdim=True)
        x = x / norm
        x_last = x.clone()

        for i in range(max_iter):
            x = x.reshape(bsz * n, n)
            x = self.forward(x, edge_index, K)
            x = x.reshape(bsz, n**2)
            norm = x.norm(p=2, dim=-1, keepdim=True)
            x = x / norm

            # print(f"{float((x - x_last).norm(p=2, dim=-1).max()):.2e}")
            if (x - x_last).norm(p=2, dim=-1).max() < tol:
                break
            x_last = x.clone()

        x = x.reshape(bsz, n, n).transpose(1, 2)
        return x


class GraphMatcher(nn.Module):
    def __init__(
        self,
        pooling_type="max",
        max_iter=1000,
        tol=1e-2,
        noise_coeff=1e-6,
        num_seed=5,
        dtype="single",
    ):
        super(GraphMatcher, self).__init__()
        if noise_coeff <= 0:
            warnings.warn("noise_coeff should be non-negative value")
            noise_coeff = 0

        if noise_coeff == 0:
            num_seed = 1

        if num_seed < 1:
            num_seed = 1

        if dtype in ["single", "float32", "float"]:
            self.dtype = torch.float32
        elif dtype in ["double", "float64"]:
            self.dtype = torch.float64
        else:
            raise ValueError(
                "dtype should be one of ['single', 'double', 'float32', 'float64']"
            )

        self.mpm = MPMModule(
            pooling_type=pooling_type,
            max_iter=max_iter,
            # tol=tol,  # HACK: REMOVED (at 241113)
            noise_coeff=noise_coeff,
            dtype=self.dtype,
        )
        self.pooling_type = pooling_type
        self.num_try = num_seed
        self.max_iter = max_iter
        self.noise_coeff = noise_coeff
        self.num_seed = num_seed
        self.tol = tol

        print(f"GraphMatchier Initialized")
        print(f"pooling_type : {self.pooling_type}")
        print(f"max_iter : {self.max_iter}")
        print(f"tol : {self.tol}")
        print(f"noise_coeff : {self.noise_coeff}")
        print(f"num_seed : {self.num_seed}")
        print(f"dtype : {self.dtype}")

        # DEBUG
        import os
        from pathlib import Path

        if (scratch_dir := Path("/scratch")).exists():
            self.scratch_dir = (
                scratch_dir / os.getenv("USER") / os.getenv("SLURM_JOBID")
            )
            self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def forward(
        self, K, edge_index, n_nodes1=None, n_nodes2=None, bsz=None, local_rank=None
    ):
        X = self.mpm.solve(
            K,
            edge_index,
            bsz=bsz,
            max_iter=self.max_iter,
            tol=self.tol,
            noise_coeff=self.noise_coeff,
        )
        X = pygm.hungarian(X.to("cpu"), n1=n_nodes1, n2=n_nodes2).to(K.device)
        perm = torch.argsort(self.padding(X.argmax(dim=-1), n_nodes1), dim=-1)
        return perm

    def padding(self, X, num_nodes, val=1e6):
        r"""
        pad each row of X at the tail with val
        each row has different size of tail which is defined by num_nodes
        X : (bsz, N)
        num_nodes : (bsz,)
        """
        max_length = X.size(1)
        indices = (
            torch.arange(max_length).expand(len(num_nodes), max_length).to(X.device)
        )
        mask = indices >= num_nodes.unsqueeze(1)

        X = torch.where(mask, torch.full_like(X, val), X)
        return X

    def solve(
        self,
        X_0,
        X_T,
        E_0,
        E_T,
        conn_0,
        conn_T,
        attr_0,
        attr_T,
        ne0,
        neT,
        bsz,
        max_num_nodes,
        nn0=None,
        nnT=None,
        max_iter=None,
        tol=None,  # HACK: NOT USED
        num_try=None,
        dtype=None,
        local_rank=None,
    ):
        r"""
        solve graph matching problem
        X_0, X_T : (bsz, n_node, n_node_feat) # both have the same number of nodes
        E_0, E_T : (bsz, n_node, n_node, n_edge_feat)
        attr_0, attr_T : (n_edge, n_edge_feat)
        ne0, neT : (n_graphs) # number of edges for each graph
        bsz <class 'int'> : number of graphs (or batch size)
        max_num_nodes <class 'int'> : the number of node in each graph. Each have the same number of nodes
        max_iter <class 'int'> : the maximum number of iteration
        tol <class 'float'> : tolerance for convergence
        num_try <class 'int'> : the number of trials
        """
        if num_try is None:
            num_try = self.num_try
        if dtype is None:
            dtype = self.dtype

        K, edge_index = construct_K(
            X_0,
            X_T,
            conn_0,
            conn_T,
            attr_0,
            attr_T,
            ne0,
            neT,
            bsz,
            max_num_nodes,
            nn0=nn0,
            nnT=nnT,
            dtype=dtype,
        )

        perm_list = []
        for _ in range(num_try):
            perm = self.forward(
                K,
                edge_index,
                n_nodes1=nn0,
                n_nodes2=nnT,
                bsz=bsz,
                local_rank=local_rank,
            )
            perm_list.append(perm)

        perm = self.select_perm(X_0, X_T, E_0, E_T, perm_list)
        nll_init = self.check_nll(X_0, X_T, E_0, E_T)
        X_0, E_0 = self.apply_perm(X_0, E_0, perm)
        nll_final = self.check_nll(X_0, X_T, E_0, E_T)

        if any(nll_init < nll_final):
            warnings.warn(
                "Some graphs are not improved by the graph matching"
                "algorithm. The original graphs are returned."
                f"total : {len(nll_init)}, failure : {torch.sum(nll_init < nll_final)}"
            )
            idx = torch.where(nll_init < nll_final)[0]
            perm[idx] = torch.arange(max_num_nodes).to(perm.device)
            nll_final = torch.min(torch.stack([nll_init, nll_final]), dim=0).values

        return perm, nll_init, nll_final

        # else:
        #     dense_0 = utils.PlaceHolder(X=X_0, E=E_0, y=None)
        #     dense_T = utils.PlaceHolder(X=X_T, E=E_T, y=None)
        #     mask_0 = torch.ones(bsz, max_num_nodes)
        #     mask_T = torch.ones(bsz, max_num_nodes)
        #     return dense_0, dense_T, mask_0, mask_T, nll_init, nll_final, perm

    def select_perm(self, X_0, X_T, E_0, E_T, perm_list):
        bsz = X_0.size(0)
        nll_list = []
        for perm in perm_list:
            X_0, E_0 = self.apply_perm(X_0, E_0, perm)
            nll = self.check_nll(X_0, X_T, E_0, E_T)
            nll_list.append(nll)

        nll_list = torch.stack(nll_list)  # (num_try, bsz)
        select_index = nll_list.argmin(dim=0)  # (bsz)

        perms = torch.stack(perm_list)
        perms = perms[select_index, torch.arange(bsz)]
        return perms

    def check_nll(self, X_0, X_T, E_0, E_T, eps=1e-18):
        bsz = X_0.size(0)
        nll_X = -torch.log(X_0 + eps) * X_T
        nll_E = -torch.log(E_0 + eps) * E_T
        nll_X = nll_X.reshape(bsz, -1).sum(dim=-1)
        nll_E = nll_E.reshape(bsz, -1).sum(dim=-1)
        nll = nll_X + nll_E
        return nll

    def apply_perm(self, X, E, perm):
        bsz = X.size(0)
        X = X[torch.arange(bsz)[:, None], perm]
        E = E[torch.arange(bsz)[:, None], perm[:, :]].transpose(1, 2)
        E = E[torch.arange(bsz)[:, None], perm[:, :]].transpose(1, 2)
        return X, E
