import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.loader import DataLoader

import ddsbm.utils as utils
from ddsbm.diffusion.distributions import DistributionNodes


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            batch_size=cfg.train.batch_size if "debug" not in cfg.general.name else 2,
            num_workers=cfg.train.num_workers,
            pin_memory=getattr(cfg.dataset, "pin_memory", False),
        )
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None
        self.batch_size = cfg.train.batch_size
        self.sample_batch_size = cfg.general.sample_batch_size

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    def node_counts(self, both, max_nodes_possible=300):
        """Modified for Bridge
        Args:
            both (bool): If True, counts both x_0 and x_T, otherwise only x_0
        """
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                unique, counts = torch.unique(data.x_T_batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
                if both:
                    unique, counts = torch.unique(data.x_0_batch, return_counts=True)
                    for count in counts:
                        all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self, both):
        """Modified for Bridge
        Args:
            both (bool): If True, counts both x_0 and x_T, otherwise only x_0
        """
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.x_T.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.train_dataloader()):
            counts += data.x_T.sum(dim=0)
            if both:
                counts += data.x_0.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self, both):
        """Modified for Bridge
        Args:
            both (bool): If True, counts both edge_attr_0 and edge_attr_T, otherwise only edge_attr_0
        """
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr_T.shape[1]

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.train_dataloader()):
            unique, counts = torch.unique(data.x_T_batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index_T.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr_T.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

            if both:
                unique, counts = torch.unique(data.x_0_batch, return_counts=True)

                all_pairs = 0
                for count in counts:
                    all_pairs += count * (count - 1)

                num_edges = data.edge_index_0.shape[1]
                num_non_edges = all_pairs - num_edges

                edge_types = data.edge_attr_0.sum(dim=0)
                assert num_non_edges >= 0
                d[0] += num_non_edges
                d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes, both):
        # Max valency possible if everything is connected
        valencies = torch.zeros(3 * max_n_nodes - 2)

        # No bond, single bond, double bond, triple bond, aromatic bond
        # No bond, single bond, double bond, triple bond, aromatic bond, two-hop bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for data in self.train_dataloader():
            n = data.x_T.shape[0]

            for atom in range(n):
                edges = data.edge_attr_T[data.edge_index_T[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1

            if both:
                n = data.x_0.shape[0]

                for atom in range(n):
                    edges = data.edge_attr_0[data.edge_index_0[0] == atom]
                    edges_total = edges.sum(dim=0)
                    valency = (edges_total * multiplier).sum()
                    valencies[valency.long().item()] += 1

        valencies = valencies / valencies.sum()
        return valencies

    # NOTE : This is function for graph augmentation
    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        r"""
        Pre-compute the input and output data's dimensions
        Update the self.input_dims and self.output_dims in place

        Args:
            datamodule (AbstractDataModule): The datamodule to use
            extra_features (callable): A function that returns the extra features
            domain_features (callable): A function that returns the domain features
        """
        # NOTE: for debugging
        __name = utils.get_current_class_and_method(self)

        example_batch = next(iter(datamodule.train_dataloader()))

        # NOTE: x_0 and x_T are the same
        k_x = "x_0"
        k_attr = "edge_attr_0"
        k_idx = "edge_index_0"
        k_batch = "x_0_batch"

        ex_dense, node_mask = utils.to_dense(
            example_batch[k_x],
            example_batch[k_idx],
            example_batch[k_attr],
            example_batch[k_batch],
        )
        example_data = {
            "X_T": ex_dense.X,
            "E_T": ex_dense.E,
            "y_T": example_batch["y"],
            "X_t": ex_dense.X,
            "E_t": ex_dense.E,
            "y_t": example_batch["y"],
            "node_mask": node_mask,
        }

        self.input_dims = {
            "X": example_batch[k_x].size(1),
            "E": example_batch[k_attr].size(1),
            "y": example_batch["y"].size(1) + 1,
        }  # + 1 due to time conditioning
        print(f"debug] {__name} \n\t input_dims INITIAL: {self.input_dims}")

        ex_extra_feat = extra_features(example_data)
        self.input_dims["X"] += ex_extra_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_feat.y.size(-1)
        print(f"debug] {__name} \n\t input_dims with EXTRA FEATURES: {self.input_dims}")

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims["X"] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_molecular_feat.y.size(-1)
        print(
            f"debug] {__name} \n\t input_dims with EXTRA DOMAIN FEATURES: {self.input_dims}"
        )

        self.output_dims = {
            "X": example_batch[k_x].size(1),
            "E": example_batch[k_attr].size(1),
            "y": 0,
        }
