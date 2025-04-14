import pygmtools as pygm
import torch
import torch.nn.functional as F
import torch_geometric


def preproc_graph_match(
    dense_0,
    dense_T,
    mask_0,
    mask_T,
    dummy_pad=False,
    dummy_index=0,
    full_edge_0=False,
    full_edge_T=False,
):
    r"""
    dense : PlaceHolder(X, E, y)
    mask : (bsz, n_node)
    X : (bsz, n_node, n_node_feat)
    E : (bsz, n_node, n_node, n_edge_feat)
    y : (bsz, n_node)
    """
    X_T = dense_T.X
    E_T = dense_T.E
    X_0 = dense_0.X
    E_0 = dense_0.E

    dev = X_0.device

    NN = X_0.size(1)
    bsz = X_0.size(0)
    nef = E_0.size(-1)
    nxf = X_0.size(-1)

    if dummy_pad:
        X_0[~mask_0.bool()] = (
            F.one_hot(
                torch.tensor(
                    dummy_index,
                ),
                num_classes=nxf,
            )
            .float()
            .to(dev)
        )
        X_T[~mask_T.bool()] = (
            F.one_hot(
                torch.tensor(
                    dummy_index,
                ),
                num_classes=nxf,
            )
            .float()
            .to(dev)
        )

    if full_edge_0:
        E_0 = E_0.argmax(dim=-1) + 1 - torch.eye(E_0.shape[1], device=dev)
        E_0 = E_0 * (mask_0.unsqueeze(2) & mask_0.unsqueeze(1)).float()
    else:
        E_0 = E_0.argmax(dim=-1)

    if full_edge_T:
        E_T = E_T.argmax(dim=-1) + 1 - torch.eye(E_T.shape[1], device=dev)
        E_T = E_T * (mask_T.unsqueeze(2) & mask_T.unsqueeze(1)).float()
    else:
        E_T = E_T.argmax(dim=-1)

    conn_0, attr_0, ne0 = pygm.utils.dense_to_sparse(E_0)
    conn_T, attr_T, neT = pygm.utils.dense_to_sparse(E_T)
    # conn: (bsz, n_edge_max, 2), attr: (bsz, n_edge_max, 1), ne: (bsz,)

    # make it sparse (remove batch dimension)
    _ = torch.arange(bsz, device=dev).unsqueeze(-1) * NN
    indices = torch.arange(conn_0.size(1), device=dev).unsqueeze(0).repeat(bsz, 1)
    mask = indices < ne0.unsqueeze(1)
    conn_0 = torch.masked_select(conn_0, mask.unsqueeze(2)).view(-1, 2)
    conn_0 += (_ * mask)[mask].unsqueeze(
        -1
    )  # make edge index increasing with batch index
    conn_0 = conn_0.t()
    attr_0 = torch.masked_select(attr_0, mask.unsqueeze(2)).view(-1, 1)
    if full_edge_0:
        attr_0 = attr_0 - 1

    indices = torch.arange(conn_T.size(1), device=dev).unsqueeze(0).repeat(bsz, 1)
    mask = indices < neT.unsqueeze(1)
    conn_T = torch.masked_select(conn_T, mask.unsqueeze(2)).view(-1, 2)
    conn_T += (_ * mask)[mask].unsqueeze(
        -1
    )  # make edge index increasing with batch index
    conn_T = conn_T.t()
    attr_T = torch.masked_select(attr_T, mask.unsqueeze(2)).view(-1, 1)
    if full_edge_T:
        attr_T = attr_T - 1

    attr_0 = F.one_hot(attr_0.squeeze(-1).long(), num_classes=nef).float()
    attr_T = F.one_hot(attr_T.squeeze(-1).long(), num_classes=nef).float()

    E_0 = dense_0.E
    E_T = dense_T.E

    nn0 = torch.ones_like(mask_0).sum(dim=1)
    nnT = torch.ones_like(mask_T).sum(dim=1)

    return_dict = {
        "X_0": X_0,
        "X_T": X_T,
        "E_0": E_0,
        "E_T": E_T,
        "conn_0": conn_0,
        "conn_T": conn_T,
        "attr_0": attr_0,
        "attr_T": attr_T,
        "ne0": ne0,
        "neT": neT,
        "nn0": nn0,
        "nnT": nnT,
    }
    return return_dict


def construct_K(
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
    nn0=None,
    nnT=None,
    dtype=None,
):
    r"""
    X_0, X_T : (bsz, n_node, n_node_feat) # both have the same number of nodes
    conn_0, conn_T : (2, n_edge) # both have different number of edges
    attr_0, attr_T : (n_edge, n_edge_feat)
    ne0 : (n_graphs,) # number of edges for each graph
    neT : (n_graphs,) # number of edges for each graph
    bsz : number of graphs (or batch size)
    max_num_nodes : the number of node in each graph.
    (Optional)
    NOTE: if nn0, nnT is not provided, it is assumed that all graphs have the same number of nodes
    nn0, nnT : (n_graphs,) number of nodes for each graph
    """
    if dtype is None:
        dtype = X_0.dtype

    dev = X_0.device
    X_0 = X_0.reshape(bsz * max_num_nodes, -1)
    X_T = X_T.reshape(bsz * max_num_nodes, -1)

    if nn0 is None or nnT is None:
        nn0 = nnT = torch.ones(bsz, device=dev).long() * max_num_nodes
        batch_0 = torch.arange(bsz, device=dev).repeat_interleave(
            max_num_nodes
        )  # batch index for each node
        batch_T = batch_0  # batch index for each node
    else:
        assert (nn0 == nnT).all()
        batch_0 = torch.arange(bsz, device=dev).repeat_interleave(nn0)
        # batch_T = torch.arange(bsz).repeat_interleave(nnT)

    ptr_0 = torch.cat([torch.LongTensor([0]).to(dev), nn0.cumsum(dim=0)])
    ptr_T = torch.cat([torch.LongTensor([0]).to(dev), nnT.cumsum(dim=0)])

    batch_edge_0 = batch_0[conn_0[0]]
    # batch_edge_T = batch_T[conn_T[0]]
    edge_ptr_0 = torch.cat([torch.LongTensor([0]).to(dev), ne0.cumsum(dim=0)])
    edge_ptr_T = torch.cat([torch.LongTensor([0]).to(dev), neT.cumsum(dim=0)])

    relative_transform = torch.cat(
        [torch.arange(ptr_0[i], ptr_0[i + 1]).to(dev) - ptr_0[i] for i in range(bsz)]
    )

    # construct K_edge
    repeat_edge_index_0 = conn_0.repeat_interleave(neT[batch_edge_0], dim=1)
    a, b = relative_transform[repeat_edge_index_0]
    eT = torch.cat(
        [
            torch.arange(edge_ptr_T[i], edge_ptr_T[i + 1], device=dev).repeat(ne0[i])
            for i in range(bsz)
        ]
    )
    e0 = torch.cat(
        [torch.arange(edge_ptr_0[i], edge_ptr_0[i + 1]).to(dev) for i in range(bsz)]
    ).repeat_interleave(neT[batch_edge_0], dim=0)

    index = torch.stack([eT, a, b])
    # val = (attr_0[e0] * attr_T[eT]).sum(dim=-1)
    val = _lazy_compute(attr_0, attr_T, e0, eT)
    if index.numel() == 0 and val.numel() == 0:
        K_edge = None
    else:
        K_edge = torch.sparse_coo_tensor(
            index, val, size=(neT.sum(), max_num_nodes, max_num_nodes), dtype=dtype
        )

    # construct K_node
    eT = torch.cat(
        [
            torch.arange(ptr_T[i], ptr_T[i + 1], device=dev).repeat_interleave(nn0[i])
            for i in range(bsz)
        ]
    )
    e0 = torch.cat(
        [
            torch.arange(ptr_0[i], ptr_0[i + 1], device=dev).repeat(nnT[i])
            for i in range(bsz)
        ]
    )
    a = relative_transform[e0]

    index = torch.stack([eT, a, a])
    # val = (X_0[e0] * X_T[eT]).sum(dim=-1)
    val = _lazy_compute(X_0, X_T, e0, eT)
    K_node = torch.sparse_coo_tensor(
        index, val, size=(nnT.sum(), max_num_nodes, max_num_nodes), dtype=dtype
    )

    self_loop_index = torch.arange(nnT.sum(), device=dev).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([conn_T, self_loop_index], dim=1)
    if K_edge is not None:
        K = torch.cat([K_edge.to_dense(), K_node.to_dense()], dim=0)
    else:
        K = K_node.to_dense()

    K = K.reshape(-1, max_num_nodes * max_num_nodes)
    return K, edge_index


def _lazy_compute(source1, source2, idx1, idx2, size=1500000):
    r"""
    compute (source1[idx1] * source2[idx2]).sum(dim=-1)
    But, it is lazy in the sense that it does not compute the whole thing at once.
    """
    # print(f"Lazy compute: {source1.size()}, {source2.size()}, {idx1.size()}, {idx2.size()}")
    if idx1.numel() == 0 or idx2.numel() == 0:
        return torch.tensor([], device=source1.device)

    for i in range(0, idx1.size(0), size):
        idx1_ = idx1[i : i + size]
        idx2_ = idx2[i : i + size]
        if i == 0:
            out = (source1[idx1_] * source2[idx2_]).sum(dim=-1)
        else:
            out = torch.cat([out, (source1[idx1_] * source2[idx2_]).sum(dim=-1)], dim=0)
    return out


def permute(X, E, perm):
    X = X[:, perm, :]
    E = E[:, perm, :, :][:, :, perm, :]
    return X, E


def compute_likelihood(M_0, M_T, q_M):
    """
    Return q(M_T|M_0) = M_0 @ q_M @ M_T.T
    """
    init_dim = len(M_0.shape)
    M_0 = M_0.flatten(start_dim=1, end_dim=-2)  # (bs, N, d) with N = n or n * n
    M_T = M_T.flatten(start_dim=1, end_dim=-2)  # (bs, N, d) with N = n or n * n

    q_M = q_M.to(M_0.dtype)
    prod = M_0 @ q_M
    # print(prod)
    likelihood = prod * M_T
    likelihood = torch.sum(likelihood, dim=-1)

    if init_dim == 3:  # Node case
        likelihood = torch.log(likelihood)
        likelihood = torch.where(
            torch.isinf(likelihood), torch.zeros_like(likelihood), likelihood
        )
        return likelihood.sum(-1)
    else:
        likelihood = torch.log(likelihood)
        likelihood = torch.where(
            torch.isinf(likelihood), torch.zeros_like(likelihood), likelihood
        )
        # Sum over all atoms except the diagonal
        # likelihood = likelihood + torch.eye(num_atoms).unsqueeze(
        #     0).flatten(start_dim=1, end_dim=-1).to(likelihood.device)
        return likelihood.sum(-1)
