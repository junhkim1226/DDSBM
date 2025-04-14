import os
import os.path as osp

import pandas as pd
import torch
import tqdm


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros(len(data.y), 0)
        return data


def SB_post_process(cfg, dataset_dir, perm_dir):
    r"""Post-process the predictions and save the results"""
    dense_data_dict = torch.load(osp.join(dataset_dir, "raw/preprocessed.pt"))
    perm_path = osp.join(perm_dir, "processed")
    file_list = os.listdir(perm_path)
    file_list = [file for file in file_list if "match_perm" in file]
    # print(f'Debug] file_list {file_list}')

    perm_dict = {}
    for file in file_list:
        perm_dict.update(torch.load(osp.join(perm_path, file)))

    perm_path = osp.join(perm_dir, "processed")
    file_list = os.listdir(perm_path)
    file_list = [file for file in file_list if "nll_df" in file]
    dfs = [pd.read_csv(osp.join(perm_path, file)) for file in file_list]
    df = pd.concat(dfs, axis=0)

    idx = torch.LongTensor(
        [(int(k[1:-1].split(",")[0]), int(k[1:-1].split(",")[1])) for k in df["key"]]
    )
    nll = df["selected_nll"].to_numpy()

    nll_matrix = torch.sparse_coo_tensor(
        idx.t(),
        torch.tensor(nll),
        size=(idx[:, 0].max() + 1, idx[:, 1].max() + 1),
    ).to_dense()
    prob_matrix = pygm.linear_solvers.sinkhorn(nll_matrix)
    save_path = osp.join(dataset_dir, "processed", "train_joint_prob.pt")
    torch.save(prob_matrix, save_path)
    # sample 5 samples from each row
    sample = torch.distributions.categorical.Categorical(prob_matrix).sample(
        sample_shape=(5,)
    )
    selected_idx = torch.stack(
        [torch.arange(prob_matrix.shape[0]).repeat(5, 1).flatten(), sample.flatten()]
    ).T
    selected_idx = [tuple([j.item() for j in i]) for i in selected_idx]

    # print(f'Debug] len(pyg_dict): {len(dense_data_dict)}')
    # print(f'Debug] len(perm_dict): {len(perm_dict)}')
    assert len(perm_dict) == len(dense_data_dict), (
        "Number of perm_dict and dense_data_dict must be the same"
    )

    data_list = []
    pre_transform = RemoveYTransform()
    # 3. Permute the data
    for ii in tqdm(range(len(dense_data_dict))):
        dense_data = dense_data_dict[ii]
        key = dense_data["key"]
        if key not in selected_idx:
            continue
        perm = perm_dict[key]
        # print(perm)

        max_num_nodes = cfg.dataset.max_num_nodes

        X_0, E_0, node_mask_0 = (
            dense_data["X_0"],
            dense_data["E_0"],
            dense_data["node_mask_0"],
        )
        X_T, E_T, node_mask_T = (
            dense_data["X_T"],
            dense_data["E_T"],
            dense_data["node_mask_T"],
        )
        X_0, E_0 = _permute(X_0, E_0, perm)
        edge_index_0, edge_attr_0 = dense_to_sparse(torch.argmax(E_0, dim=-1))
        edge_attr_0 = F.one_hot(edge_attr_0, num_classes=E_0.size(-1))
        edge_index_T, edge_attr_T = dense_to_sparse(torch.argmax(E_T, dim=-1))
        edge_attr_T = F.one_hot(edge_attr_T, num_classes=E_T.size(-1))

        # if not cfg.dataset.name == "molecule":
        data = utils.PairData(
            x_0=X_0.squeeze(0),
            x_T=X_T.squeeze(0),
            edge_index_0=edge_index_0,
            edge_index_T=edge_index_T,
            edge_attr_0=edge_attr_0,
            edge_attr_T=edge_attr_T,
            num_nodes=node_mask_0.sum(),
            y=torch.ones(
                1,
            ),
            idx=key,
        )

        # else:
        #     data = utils.PairData(x_0=X_0.squeeze(0),
        #                           x_T=X_T.squeeze(0),
        #                           edge_index_0=edge_index_0,
        #                           edge_index_T=edge_index_T,
        #                           edge_attr_0=edge_attr_0,
        #                           edge_attr_T=edge_attr_T,
        #                           num_nodes=pyg_data.num_nodes,
        #                           smi_0=pyg_data.smi_0,
        #                           smi_T=pyg_data.smi_T,
        #                           y=pyg_data.y,
        #                           idx=key)

        if pre_transform is not None:
            data = pre_transform(data)
        data_list.append(data)

    # TODO: Change directory
    save_path = osp.join(dataset_dir, "processed", "train_data.pt")
    torch.save(data_list, save_path + ".tmp")

    print(f"Debug] {save_path}")
    col = _collate(data_list)
    torch.save(_collate(data_list), save_path)


def post_process(cfg, dataset_dir, perm_dir):
    r"""Post-process the predictions and save the results"""
    if cfg.graph_match.initialize:
        pyg_data_dict = torch.load(osp.join(dataset_dir, "train_data.pt"))

        perm_path = osp.join(perm_dir, "processed")
        file_list = os.listdir(perm_path)
        file_list = [file for file in file_list if "match_perm" in file]
        print(f"Debug] file_list {file_list}")

        perm_dict = {}
        for file in file_list:
            perm_dict.update(torch.load(osp.join(perm_path, file)))

    else:
        file_list = os.listdir(cfg.graph_match.data_path)
        file_list = [file for file in file_list if "mol" not in file]
        file_list = [file for file in file_list if "generated_joint_train" in file]
        pyg_data_dict = {}
        for file in file_list:
            pyg_data_dict.update(torch.load(osp.join(cfg.graph_match.data_path, file)))

        # 2. Load perm data
        perm_path = osp.join(perm_dir, "processed")
        file_list = os.listdir(perm_path)
        file_list = [file for file in file_list if "match_perm" in file]

        perm_dict = {}
        for file in file_list:
            perm_dict.update(torch.load(osp.join(perm_path, file)))

    print(f"Debug] len(pyg_dict): {len(pyg_data_dict)}")
    print(f"Debug] len(perm_dict): {len(perm_dict)}")
    assert len(perm_dict) == len(pyg_data_dict), (
        "Number of perm_dict and pyg_data_dict must be the same"
    )

    data_list = []
    pre_transform = RemoveYTransform()

    # 3. Permute the data
    for key in tqdm(perm_dict.keys()):
        pyg_data = pyg_data_dict[key]
        perm = perm_dict[key]

        max_num_nodes = cfg.dataset.max_num_nodes

        dense_data_0_p, dense_data_T_p, node_mask_0_p, node_mask_T_p = (
            utils._to_dense_pair(pyg_data, max_num_nodes=max_num_nodes)
        )

        X_0, E_0 = dense_data_0_p.X, dense_data_0_p.E
        X_0, E_0 = permute(X_0, E_0, perm)
        edge_index_0, edge_attr_0 = dense_to_sparse(torch.argmax(E_0, dim=-1))
        edge_attr_0 = F.one_hot(edge_attr_0, num_classes=E_0.size(-1))

        if cfg.dataset.name in ["planar", "sbm", "protein", "comm20"]:
            data = utils.PairData(
                x_0=X_0.squeeze(0),
                x_T=pyg_data.x_T,
                edge_index_0=edge_index_0,
                edge_index_T=pyg_data.edge_index_T,
                edge_attr_0=edge_attr_0,
                edge_attr_T=pyg_data.edge_attr_T,
                num_nodes=pyg_data.num_nodes,
                y=pyg_data.y,
                idx=pyg_data.idx,
            )

        else:
            data = utils.PairData(
                x_0=X_0.squeeze(0),
                x_T=pyg_data.x_T,
                edge_index_0=edge_index_0,
                edge_index_T=pyg_data.edge_index_T,
                edge_attr_0=edge_attr_0,
                edge_attr_T=pyg_data.edge_attr_T,
                num_nodes=pyg_data.num_nodes,
                smi_0=pyg_data.smi_0,
                smi_T=pyg_data.smi_T,
                y=pyg_data.y,
                idx=pyg_data.idx,
            )
        if pre_transform is not None:
            data = pre_transform(data)
        data_list.append(data)

    # TODO: Change directory
    save_path = osp.join(dataset_dir, "processed", "train_data.pt")

    print(f"Debug] {save_path}")
    col = _collate(data_list)
    torch.save(_collate(data_list), save_path)
