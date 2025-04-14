import glob
import os
import os.path as osp
import pathlib
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import rdkit
import torch
import torch.nn.functional as F
import torch_geometric
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

import ddsbm.utils as utils
from ddsbm.analysis.rdkit_functions import (
    build_molecule_with_partial_charges,
    compute_molecular_metrics,
    mol2smiles,
)
from ddsbm.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from ddsbm.diffusion import diffusion_utils

RDLogger.DisableLog("rdApp.*")


class RemoveYTransform:
    def __call__(self, data):
        # print(f"[Debug] (RemoveYTransform) {data}")
        data.y = torch.zeros(len(data.y), 0)
        return data


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def process_mol(mol, smi, max_num_nodes, atom_decoder):
    types = {atom: i for i, atom in enumerate(atom_decoder)}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    N = mol.GetNumAtoms()

    type_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])

    # NOTE : Padding with max_num_nodes
    while len(type_idx) < max_num_nodes:
        type_idx.append(types["X"])
    assert len(type_idx) == max_num_nodes, f"{smi}, {len(type_idx)}, {max_num_nodes}"

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()] + 1]

    if len(row) == 0:
        return None

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
    y = torch.zeros(size=(1, 0), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smi=smi)

    return data


# NOTE : Add for prior data generation (junhkim)
def sample_prior_graphs(
    num_nodes, limit_dist, batch_size=1, device=torch.device("cpu")
):
    """Sample prior graphs from limit_dist.
    Args:
        num_nodes (int or torch.Tensor): Number of nodes to sample for each graph.
        node_dist (np.array): Distribution of the number of nodes.
        limit_dist (utils.Placeholder): Distribution of the node types, edge types.
        batch_size (int): Number of graphs to sample.
        device (torch.device): Device to put the tensors on.
    Returns:
        torch.Tensor: The node features.
        torch.Tensor: The edge index.
        torch.Tensor: The edge attributes.
    """
    n_nodes = num_nodes * torch.ones(batch_size, device=device, dtype=torch.int)
    n_max = torch.max(n_nodes).item()
    # Build the masks
    arange = torch.arange(n_max, device=device).unsqueeze(0).expand(batch_size, -1)
    node_mask = arange < n_nodes.unsqueeze(1)
    # Sample noise  -- z has size (n_samples, n_nodes, n_features)
    z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist, node_mask=node_mask)
    X, E, y = z_T.X, z_T.E, z_T.y

    X = X.squeeze(0)
    E = E.squeeze(0).argmax(dim=-1)
    y = y.squeeze(0)

    edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(E)
    edge_attr = F.one_hot(edge_attr, num_classes=limit_dist.E.shape[-1]).to(torch.float)
    # edge_attr = torch.zeros(edge_index.shape[-1], E.shape[-1], dtype=torch.float)
    # edge_attr[:, 1] = 1
    return X, edge_index, edge_attr


class PlanarDataset(InMemoryDataset):
    def __init__(
        self,
        stage,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        max_num_nodes=30,
        compute_dataset_infos=False,
        prior_sample=False,
        transition="uniform",
    ):
        self.max_num_nodes = max_num_nodes
        self.stage = stage

        # NOTE : For unconditional generation
        self.compute_dataset_infos = compute_dataset_infos
        self.prior_sample = prior_sample
        if not self.compute_dataset_infos:
            node_types = torch.load(root / "node_types.pt")
            edge_types = torch.load(root / "edge_types.pt")
            Xdim = len(node_types)
            Edim = len(edge_types)
            ydim = 12
            if transition == "uniform":
                x_limit = torch.ones(Xdim) / Xdim
                e_limit = torch.ones(Edim) / Edim
                y_limit = torch.ones(ydim) / ydim
                self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
            elif transition == "marginal":
                x_limit = node_types / torch.sum(node_types)
                e_limit = edge_types / torch.sum(edge_types)
                y_limit = torch.ones(ydim) / ydim
                self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        if self.stage == "train":
            self.file_idx = 0
        elif self.stage == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ["train_spectre.pt", "val_spectre.pt", "test_spectre.pt"]

    @property
    def split_file_name(self):
        return ["train_spectre.pt", "val_spectre.pt", "test_spectre.pt"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ["train_data.pt", "valid_data.pt", "test_data.pt"]

    def process(self):
        # NOTE : Hardcoded maximum atoms
        max_num_nodes = self.max_num_nodes
        print("MAX NUM NODES", max_num_nodes)
        print(f"Current Dataset Directory : {self.split_paths[self.file_idx]}")

        target_data = torch.load(self.split_paths[self.file_idx])

        data_list = []
        for i, adj in enumerate(target_data):
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)

            y = torch.zeros((1, 0), dtype=torch.float)
            data_T = Data(
                x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, smi="planar"
            )

            if self.compute_dataset_infos:
                data_0 = data_T
            elif self.prior_sample:
                # TODO : Implement sample_prior_graphs
                X_0, edge_index_0, edge_attr_0 = sample_prior_graphs(
                    num_nodes=data_T.num_nodes,
                    limit_dist=self.limit_dist,
                    batch_size=1,
                    device="cpu",
                )
                y = torch.zeros(size=(1, 0), dtype=torch.float)
                data_0 = Data(
                    x=X_0,
                    edge_index=edge_index_0,
                    edge_attr=edge_attr_0,
                    y=y,
                    smi="prior",
                )

            data = utils.PairData(
                x_0=data_0.x,
                x_T=data_T.x,
                x_0_p=data_0.x,
                x_T_p=data_T.x,
                edge_index_0=data_0.edge_index,
                edge_index_T=data_T.edge_index,
                edge_index_0_p=data_0.edge_index,
                edge_index_T_p=data_T.edge_index,
                edge_attr_0=data_0.edge_attr,
                edge_attr_T=data_T.edge_attr,
                edge_attr_0_p=data_0.edge_attr,
                edge_attr_T_p=data_T.edge_attr,
                smi_0=data_0.smi,
                smi_T=data_T.smi,
                y=y,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        print("Save Collated data into", self.processed_paths[self.file_idx])
        data_dict = dict([(d["idx"], d) for d in data_list])
        raw_path = self.processed_paths[self.file_idx].split("processed")[0]
        raw_path = osp.join(raw_path, self.processed_file_names[self.file_idx])
        torch.save(data_dict, raw_path)
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


class PlanarDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.data_dir = cfg.dataset.datadir
        self.remove_h = False

        self.base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        self.root_path = self.base_path / self.data_dir

        self.max_num_nodes = 64

        # NOTE : For Marginal Limit prior distribution
        self.compute_dataset_infos = cfg.dataset.compute_dataset_infos
        self.prior_sample = cfg.dataset.prior_sample
        self.transition = cfg.model.transition

        print("Load Initial data")
        datasets = {
            "train": PlanarDataset(
                stage="train",
                root=self.root_path,
                transform=RemoveYTransform(),
                max_num_nodes=self.max_num_nodes,
                compute_dataset_infos=self.compute_dataset_infos,
                prior_sample=self.prior_sample,
                transition=self.transition,
            ),
            "val": PlanarDataset(
                stage="val",
                root=self.root_path,
                transform=RemoveYTransform(),
                max_num_nodes=self.max_num_nodes,
                compute_dataset_infos=self.compute_dataset_infos,
                prior_sample=self.prior_sample,
                transition=self.transition,
            ),
            "test": PlanarDataset(
                stage="test",
                root=self.root_path,
                transform=RemoveYTransform(),
                max_num_nodes=self.max_num_nodes,
                compute_dataset_infos=self.compute_dataset_infos,
                prior_sample=self.prior_sample,
                transition=self.transition,
            ),
        }
        super().__init__(cfg, datasets)

    def _load_max_num_nodes(self) -> int:
        # NOTE : Load max_num_atoms
        assert (self.root_path / "max_num_atoms.txt").exists(), (
            f"{self.root_path}/max_num_atoms.txt does not exist."
        )
        "Run `get_dist_info.py` first."
        with (self.root_path / "max_num_atoms.txt").open("r") as f:
            max_num_nodes = int(f.read())
        return max_num_nodes

    def _load_atom_info(self) -> Tuple[Dict[int, float], List[str], Dict[str, int]]:
        # NOTE : Load atom encoder, decoder
        assert (self.root_path / "atom2weight.txt").exists(), (
            f"Path {self.root_path}/atom2weight.txt does not exist."
        )
        "Run `get_dist_info.py` first."
        atom_decoder = ["X"]
        atom_weights = [0.0]
        with (self.root_path / "atom2weight.txt").open("r") as f:
            for line in f:
                atom, weight = line.split()
                atom = atom.replace(":", "")
                atom_decoder.append(atom)
                atom_weights.append(float(weight))
        atom_encoder = {atom: i for i, atom in enumerate(atom_decoder)}
        atom_weights = {
            idx + 1: atom_weights[idx] for idx, atom in enumerate(atom_decoder)
        }
        return atom_weights, atom_encoder, atom_decoder

    def _load_max_weight(self) -> float:
        # NOTE : Load max_weight
        assert (self.root_path / "max_mw.txt").exists(), (
            f"Path {self.root_path}/max_mw.txt does not exist."
        )
        "Run `get_dist_info.py` first."
        with (self.root_path / "max_mw.txt").open("r") as f:
            max_weight = float(f.read().strip())
        return max_weight

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            follow_batch=["x_0", "x_T", "x_0_p", "x_T_p"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            follow_batch=["x_0", "x_T", "x_0_p", "x_T_p"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.sample_batch_size,
            follow_batch=["x_0", "x_T", "x_0_p", "x_T_p"],
        )

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.sample_batch_size,
            follow_batch=["x_0", "x_T", "x_0_p", "x_T_p"],
        )


class Planarinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, bridge=True):
        root_path = datamodule.root_path

        # should be the same as one in `process_mol`
        self.remove_h = cfg.dataset.remove_h
        # NOTE: THESE ARE PREDEFINED IN DATAMODULE
        self.max_num_nodes = datamodule.max_num_nodes

        # TODO: check atomwise property is corretly set.

        if cfg.dataset.compute_dataset_infos:
            np.set_printoptions(suppress=True, precision=5)
            # NOTE: since we pad all atoms, n_nodes distribution is 1 at max_num_nodes
            self.n_nodes = datamodule.node_counts(both=False)
            _path = root_path / "n_counts.pt"
            print(f"Debug] Saving n_counts to {_path}")
            torch.save(self.n_nodes, _path)

            self.node_types = datamodule.node_types(both=False)
            _path = root_path / "node_types.pt"
            print(f"Debug] Saving node_types to {_path}")
            torch.save(self.node_types, _path)

            self.edge_types = datamodule.edge_counts(both=False)
            _path = root_path / "edge_types.pt"
            print(f"Debug] Saving edge_types to {_path}")
            torch.save(self.edge_types, _path)

            # valencies = datamodule.valency_count(self.max_num_nodes, both=False)
            # _path = root_path / "valencies.pt"
            # print(f"Debug] Saving valencies to {_path}")
            # torch.save(valencies, _path)
            # self.valency_distribution = valencies

            # NOTE : To generate prior data for uncond (junhkim)
            print(f"Debug] Remove Temporary pt files")
            _path = root_path / "processed"

            print(f"Debug] Current processed glob")
            print("\n".join(glob.glob(str(_path) + "/*.pt")))

            for f in glob.glob(str(_path) + "/*.pt"):
                # Remove f
                os.system(f"rm {f}")

            print(f"Debug] After processed glob")
            print("\n".join(glob.glob(str(_path) + "/*.pt")))

            assert False, "statistics done"
        elif cfg.dataset.prior_sample:
            assert False, "sample done"

        self.n_nodes = torch.load(root_path / "n_counts.pt")
        self.node_types = torch.load(root_path / "node_types.pt")
        self.edge_types = torch.load(root_path / "edge_types.pt")

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)


def compute_train_smiles(atom_decoder, train_dataloader, remove_h):
    print(f"\tConverting dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(
            data.x_0, data.edge_index_0, data.edge_attr_0, data.x_0_batch
        )
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            # NOTE : Masking dummy atom
            mask = atom_types != atom_decoder.index("X")
            atom_types = atom_types[mask]
            edge_types = edge_types[mask][:, mask]

            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(
                molecule[0], molecule[1], atom_decoder
            )
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1
            print("Smiles", smile)

        if i % 1000 == 0:
            print(
                "\tConverting grambow dataset to SMILES {0:.2%}".format(
                    float(i) / len_train
                )
            )
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert dataset_infos is not None, (
            "If wanting to evaluate dataset, need to pass dataset_infos"
        )
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = "train_smiles.npy"
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_train_smiles(atom_decoder, train_dataloader, remove_h)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print(
            "Evaluating the dataset -- number of molecules to evaluate",
            len(all_molecules),
        )
        metrics = compute_molecular_metrics(
            molecule_list=all_molecules,
            train_smiles=train_smiles,
            dataset_info=dataset_infos,
        )
        print(metrics[0])

    return train_smiles
