import os
import os.path as osp
import pathlib
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
from ddsbm.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule

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


class JointMolDataset(InMemoryDataset):
    def __init__(
        self,
        stage,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        max_num_nodes=30,
    ):
        self.max_num_nodes = max_num_nodes
        self.stage = stage
        if self.stage == "train":
            self.file_idx = 0
        elif self.stage == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2

        assert (pathlib.Path(root) / "atom2weight.txt").exists(), (
            f"Path {root}/atom2weight.txt does not exist."
        )

        atom_decoder = ["X"]
        atom_weights = [0.0]
        with (root / "atom2weight.txt").open("r") as f:
            for line in f:
                atom, weight = line.split()
                atom = atom.replace(":", "")
                atom_decoder.append(atom)
                atom_weights.append(float(weight))
        self.atom_decoder = atom_decoder

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ["train.csv", "val.csv", "test.csv"]

    @property
    def split_file_name(self):
        return ["train.csv", "val.csv", "test.csv"]

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
        print(f"Current Dataset Path : {self.split_paths[self.file_idx]}")

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)

        data_list = []
        for i, line in target_df.iterrows():
            smi_0, smi_T = line["REF-SMI"], line["PRB-SMI"]
            mol_0, mol_T = Chem.MolFromSmiles(smi_0), Chem.MolFromSmiles(smi_T)
            Chem.SanitizeMol(mol_0), Chem.SanitizeMol(mol_T)
            smi_0, smi_T = Chem.MolToSmiles(mol_0), Chem.MolToSmiles(mol_T)

            data_0 = process_mol(mol_0, smi_0, max_num_nodes, self.atom_decoder)
            data_T = process_mol(mol_T, smi_T, max_num_nodes, self.atom_decoder)
            try:
                data_0 = process_mol(mol_0, smi_0, max_num_nodes, self.atom_decoder)
                data_T = process_mol(mol_T, smi_T, max_num_nodes, self.atom_decoder)
            except:
                continue

            if data_0 is None or data_T is None:
                continue

            y = torch.zeros((1, 0), dtype=torch.float)

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


class JointMolDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.data_dir = cfg.dataset.datadir
        self.remove_h = False

        self.base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        self.root_path = self.base_path / self.data_dir

        self.max_num_nodes = self._load_max_num_nodes()
        self.atom_weights, self.atom_encoder, self.atom_decoder = self._load_atom_info()
        self.max_weight = self._load_max_weight()

        print("Load Initial data")
        datasets = {
            "train": JointMolDataset(
                stage="train",
                root=self.root_path,
                transform=RemoveYTransform(),
                max_num_nodes=self.max_num_nodes,
            ),
            "val": JointMolDataset(
                stage="val",
                root=self.root_path,
                transform=RemoveYTransform(),
                max_num_nodes=self.max_num_nodes,
            ),
            "test": JointMolDataset(
                stage="test",
                root=self.root_path,
                transform=RemoveYTransform(),
                max_num_nodes=self.max_num_nodes,
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


class JointMolecularinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, bridge=True):
        root_path = datamodule.root_path

        # should be the same as one in `process_mol`
        self.remove_h = cfg.dataset.remove_h
        # NOTE: THESE ARE PREDEFINED IN DATAMODULE
        self.max_weight = datamodule.max_weight
        self.max_num_nodes = datamodule.max_num_nodes
        self.atom_weights = datamodule.atom_weights
        self.atom_encoder = datamodule.atom_encoder
        self.atom_decoder = datamodule.atom_decoder

        # TODO: check atomwise property is corretly set.
        periodic_table = Chem.GetPeriodicTable()
        self.valencies = [0] + [
            periodic_table.GetDefaultValence(atom) for atom in self.atom_decoder[1:]
        ]
        self.num_atom_types = len(self.atom_decoder)

        print(f"Debug] {datamodule}")
        print(f"Atom Decoder : {self.atom_decoder}")
        print(f"Valencies : {self.valencies}")
        print(f"Atom Weights : {self.atom_weights}")
        print(f"Debug] {datamodule}")

        if cfg.dataset.compute_dataset_infos:
            np.set_printoptions(suppress=True, precision=5)
            # NOTE: since we pad all atoms, n_nodes distribution is 1 at max_num_nodes
            self.n_nodes = datamodule.node_counts(both=True)
            _path = root_path / "n_counts.pt"
            print(f"Debug] Saving n_counts to {_path}")
            torch.save(self.n_nodes, _path)

            self.node_types = datamodule.node_types(both=True)
            _path = root_path / "node_types.pt"
            print(f"Debug] Saving node_types to {_path}")
            torch.save(self.node_types, _path)

            self.edge_types = datamodule.edge_counts(both=True)
            _path = root_path / "edge_types.pt"
            print(f"Debug] Saving edge_types to {_path}")
            torch.save(self.edge_types, _path)

            valencies = datamodule.valency_count(self.max_num_nodes, both=True)
            _path = root_path / "valencies.pt"
            print(f"Debug] Saving valencies to {_path}")
            torch.save(valencies, _path)
            self.valency_distribution = valencies
            assert False, "statistics done"

        self.n_nodes = torch.load(root_path / "n_counts.pt")
        self.node_types = torch.load(root_path / "node_types.pt")
        self.edge_types = torch.load(root_path / "edge_types.pt")
        self.valency_distribution = torch.load(root_path / "valencies.pt")

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)


def compute_train_smiles(atom_decoder, train_dataloader, remove_h, direction):
    print(f"\tConverting dataset to SMILES for remove_h={remove_h}...")
    print(f"TRAIN SMILES DIRECTION: {direction}")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        if direction == "forward":
            dense_data, node_mask = utils.to_dense(
                data.x_T, data.edge_index_T, data.edge_attr_T, data.x_T_batch
            )
        elif direction == "backward":
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


def get_train_smiles(
    cfg,
    train_dataloader,
    dataset_infos,
    evaluate_dataset=False,
    direction: str = "forward",
):
    assert direction in ["forward", "backward"]

    if evaluate_dataset:
        assert dataset_infos is not None, (
            "If wanting to evaluate dataset, need to pass dataset_infos"
        )
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = f"train_smiles_{direction}.npy"
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_train_smiles(
            atom_decoder, train_dataloader, remove_h, direction
        )
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            if direction == "forward":
                dense_data, node_mask = utils.to_dense(
                    data.x_T, data.edge_index_T, data.edge_attr_T, data.x_T_batch
                )
            elif direction == "backward":
                dense_data, node_mask = utils.to_dense(
                    data.x_0, data.edge_index_0, data.edge_attr_0, data.x_0_batch
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
