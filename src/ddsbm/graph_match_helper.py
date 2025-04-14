import argparse
import os
import os.path as osp
import pathlib
import warnings
from collections import defaultdict

import pandas as pd
import pygmtools as pygm
import pytorch_lightning as pl
import torch  # pytorch backend
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.distributed import group as _group
from torch_geometric.data.collate import collate
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from ddsbm import utils
from ddsbm.datasets.jointmol_dataset import JointMolDataModule, JointMolecularinfos
from ddsbm.diffusion.noise_schedule import (
    DiscreteUniformTransition,
    MarginalUniformTransition,
    SymmetricNoiseScheduleDiscrete,
)

pygm.set_backend("pytorch")  # set default backend for pygmtools
warnings.filterwarnings("ignore", category=PossibleUserWarning)


from ddsbm.graph_match.module import GraphMatcher
from ddsbm.graph_match.utils import permute, preproc_graph_match


class GraphMatchModule(pl.LightningModule):
    def __init__(self, cfg, dataset_dir):
        r"""Graph matching model with Message Passing Module

        Args:
            cfg : configuration file
            dataset_dir (str) : directory of the dataset
        """
        super().__init__()
        assert cfg.graph_match.num_seed > 0, "num_seed must be greater than 0"

        self.cfg = cfg
        self.is_unconditional = cfg.dataset.unconditional
        self.full_edge_0 = cfg.graph_match.full_edge_0
        self.full_edge_T = cfg.graph_match.full_edge_T

        self.graph_matcher = GraphMatcher(
            pooling_type=cfg.graph_match.pooling_type,
            max_iter=cfg.graph_match.max_iter,
            tol=cfg.graph_match.tol,
            noise_coeff=cfg.graph_match.noise_coeff,
            dtype=cfg.graph_match.dtype,
            num_seed=cfg.graph_match.num_seed,
        )

        # Transition Model Define
        self.noise_schedule = SymmetricNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule,
            timesteps=cfg.model.diffusion_steps,
            min_alpha=cfg.model.min_alpha,
        )

        # Debug Options
        self.self_recover = cfg.graph_match.self_recover
        if self.self_recover:
            warnings.warn(f"Debug] self_recover is set to {self.self_recover}" * 100)

        if cfg.model.transition == "uniform":
            # NOTE: Number of classes (including X)
            Xdim = len(torch.load(osp.join(dataset_dir, "node_types.pt")))
            Edim = len(torch.load(osp.join(dataset_dir, "edge_types.pt")))
            self.transition_model = DiscreteUniformTransition(
                x_classes=Xdim, e_classes=Edim, y_classes=1
            )
        elif cfg.model.transition == "marginal":
            node_types = torch.load(osp.join(dataset_dir, "node_types.pt"))
            x_marginals = node_types / torch.sum(node_types)

            edge_types = torch.load(osp.join(dataset_dir, "edge_types.pt"))
            e_marginals = edge_types / torch.sum(edge_types)
            # y_marginals = torch.ones(1)

            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals, e_marginals=e_marginals, y_classes=1
            )
        else:
            raise ValueError(
                "transition should be selected between uniform and marginal"
            )

        self.predict_step_outputs = []

    def apply_transition(self, preproc_dict):
        X_0, E_0, attr_0 = (
            preproc_dict["X_0"],
            preproc_dict["E_0"],
            preproc_dict["attr_0"],
        )
        E_0 = E_0.to(X_0.dtype)
        T_float = torch.ones(1, 1).float()
        alpha_T_bar = self.noise_schedule.get_alpha_bar(t_normalized=T_float)
        QTb = self.transition_model.get_Qt_bar(alpha_T_bar, device=X_0.device)

        bsz, max_num_nodes, _ = X_0.shape
        X_0 = utils.compute_transition(X_0, QTb.X)
        E_0 = utils.compute_transition(E_0, QTb.E).reshape(
            bsz, max_num_nodes, max_num_nodes, -1
        )
        attr_0 = utils.compute_transition(attr_0.unsqueeze(0), QTb.E).squeeze(0)

        # update preproc_dict
        preproc_dict.update(
            {
                "X_0": X_0,
                "E_0": E_0,
                "attr_0": attr_0,
                "bsz": bsz,
                "max_num_nodes": max_num_nodes,
            }
        )

        for k, v in preproc_dict.items():
            if isinstance(v, torch.Tensor):
                preproc_dict[k] = v.to(device=self.device)
        preproc_dict["local_rank"] = self.local_rank
        return

    def seed_everything(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def predict_step(self, data, i):
        r"""Graph matching prediction with Max Pooling Match

        Args:
            data (PairDataBatch) : data with sparse format. All graphs are padded to the same size.
            i (int): index of data

        Returns:
            results (list): list of tuples (key, perm)
        """
        # 1. prepare dense format input
        dense_0, mask_0 = utils.to_dense(
            data.x_0_p, data.edge_index_0_p, data.edge_attr_0_p, data.x_0_p_batch
        )

        if self.is_unconditional:
            # FIXME : Debugging
            # mask_0 = torch.ones_like(dense_0.X[:, :, 0]).bool()
            mask_0 = mask_0
        else:
            mask_0 = dense_0.X.argmax(-1) != 0
        if not self.self_recover:
            dense_T, mask_T = utils.to_dense(
                data.x_T_p, data.edge_index_T_p, data.edge_attr_T_p, data.x_T_p_batch
            )
            if self.is_unconditional:
                mask_T = torch.ones_like(dense_T.X[:, :, 0]).bool()
            else:
                mask_T = dense_T.X.argmax(-1) != 0
        else:
            dense_T, mask_T = utils.to_dense(
                data.x_0_p, data.edge_index_0_p, data.edge_attr_0_p, data.x_0_p_batch
            )
            randperm = torch.arange(mask_0.shape[1])
            randperm[: mask_0.sum(-1).min()] = torch.randperm(mask_0.sum(-1).min())
            dense_T.X = dense_T.X[:, randperm, :]
            dense_T.E = dense_T.E[:, randperm, :, :][:, :, randperm, :]

        # 2. preprocess the data for graph-matching (sparsification)
        preproc_dict = preproc_graph_match(
            dense_0,
            dense_T,
            mask_0,
            mask_T,
            full_edge_0=self.full_edge_0,
            full_edge_T=self.full_edge_T,
        )

        # 3. apply transition
        self.apply_transition(preproc_dict)
        assert preproc_dict.get("bsz") and preproc_dict.get("max_num_nodes")

        if not self.cfg.graph_match.only_compute_nll:
            perm, init_nll, selected_nll = self.graph_matcher.solve(**preproc_dict)

            output = {
                "keys": data.idx,
                "perm": perm,
                "init_nll": init_nll,
                "selected_nll": selected_nll,
            }
            if self.self_recover:
                output["randperm"] = randperm

                # NOTE: REPEAT FOR IDENTITY PERMUTATION
                dense_0, mask_0 = utils.to_dense(
                    data.x_0_p,
                    data.edge_index_0_p,
                    data.edge_attr_0_p,
                    data.x_0_p_batch,
                )
                mask_0 = dense_0.X.argmax(-1) != 0
                dense_T, mask_T = utils.to_dense(
                    data.x_0_p,
                    data.edge_index_0_p,
                    data.edge_attr_0_p,
                    data.x_0_p_batch,
                )
                mask_T = dense_T.X.argmax(-1) != 0

                # 2. preprocess the data for graph-matching (sparsification)
                preproc_dict = preproc_graph_match(
                    dense_0,
                    dense_T,
                    mask_0,
                    mask_T,
                    full_edge_0=self.full_edge_0,
                    full_edge_T=self.full_edge_T,
                )

                # 3. apply transition
                self.apply_transition(preproc_dict)
                assert preproc_dict.get("bsz") and preproc_dict.get("max_num_nodes")

                # 4. graph matching
                identity_perm, identity_init_nll, identity_selected_nll = (
                    self.graph_matcher.solve(**preproc_dict)
                )

                output.update(
                    {
                        "identity_perm": identity_perm,
                        "identity_init_nll": identity_init_nll,
                        "identity_selected_nll": identity_selected_nll,
                    }
                )
        else:
            self.print("INSIDE COMPUTE NLL " * 10)
            base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
            dataset_dir = base_path / self.cfg.dataset.datadir
            output_dir = dataset_dir / "processed"
            assert output_dir.exists(), f"output_dir {output_dir} does not exist"

            iteration = self.cfg.graph_match.get("iteration", None)
            iteration_suffix = "" if iteration is None else f"_{iteration}"
            direction = self.cfg.train.get("bridge_direction", None)
            # NOTE: direction is set as forward by default
            direction_suffix = "" if iteration is None else f"_{direction}"
            # Epoch (if exists)
            epoch_suffix = ""
            seed_suffix = ""
            # NOTE: If TRUE, WE ARE DOING TEST
            if (ckpt_path := getattr(self.cfg.general, "test_only", None)) is not None:
                ckpt_path = pathlib.Path(ckpt_path)
                epoch = ckpt_path.stem.split("=")[-1]
                epoch_suffix = f"_{epoch}"
                seed_suffix = f"_seed{self.cfg.general.seed}"

            train_or_test = "train" if not self.cfg.graph_match.test else "test"
            perm_pt_path = (
                output_dir
                / f"{train_or_test}_match_perm{direction_suffix}{iteration_suffix}{epoch_suffix}{seed_suffix}.pt"
            )
            self.print("Debug] perm_pt_path")
            assert perm_pt_path.exists(), f"perm_pt_path {perm_pt_path} does not exist"

            prev_perm_dict = torch.load(perm_pt_path)
            perm = []
            for key in data.idx:
                _perm = prev_perm_dict[key.item()]
                _perm = torch.from_numpy(_perm)
                assert _perm.ndim == 1 and _perm.size() == torch.Size([37])
                perm.append(_perm)

            perm = torch.stack(perm, dim=0)
            perm = perm.to(self.device)

            nll_init = self.graph_matcher.check_nll(
                preproc_dict["X_0"],
                preproc_dict["X_T"],
                preproc_dict["E_0"],
                preproc_dict["E_T"],
            )

            X_0, E_0 = self.graph_matcher.apply_perm(
                preproc_dict["X_0"], preproc_dict["E_0"], perm
            )

            nll_final = self.graph_matcher.check_nll(
                X_0,
                preproc_dict["X_T"],
                E_0,
                preproc_dict["E_T"],
            )

            output = {
                "keys": data.idx,
                "init_nll": nll_init,
                "selected_nll": nll_final,
            }
        self.predict_step_outputs.append(output)
        return output

    def on_predict_epoch_end(self) -> None:
        list_gather_step_outputs = self._gather_objects(obj=self.predict_step_outputs)
        if not self.trainer.is_global_zero:
            return
        else:  # global zero only
            assert len(list_gather_step_outputs) == self.trainer.world_size

            predictions = []
            for outputs in list_gather_step_outputs:
                predictions.extend(outputs)  # NOTE: batched data list

            base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
            dataset_dir = base_path / self.cfg.dataset.datadir
            output_dir = dataset_dir / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)

            self.print(f"Debug] Writing predictions to {output_dir}")

            if not self.cfg.graph_match.only_compute_nll:
                perm_dict = {}
                randperm_dict = {}
                nll_dict = defaultdict(list)
                identity_nll_dict = defaultdict(list)
                for prediction in predictions:
                    keys, perm_tensor, init_nll, selected_nll = (
                        prediction["keys"],
                        prediction["perm"],
                        prediction["init_nll"],
                        prediction["selected_nll"],
                    )
                    for key, perm, init, selected in zip(
                        keys, perm_tensor, init_nll, selected_nll
                    ):
                        if key.numel() == 1:
                            k = key.item()
                        else:
                            k = tuple(key.tolist())
                        perm_dict[k] = perm.cpu().numpy()
                        nll_dict["key"].append(k)
                        nll_dict["init_nll"].append(init.item())
                        nll_dict["selected_nll"].append(selected.item())

                        if self.self_recover:
                            # NOTE: WE USE BS=1 -> randperm is shape of N
                            randperm_dict[k] = prediction["randperm"].cpu().numpy()
                            identity_nll_dict["key"].append(k)
                            identity_nll_dict["init_nll"].append(
                                prediction["identity_init_nll"].item()
                            )
                            identity_nll_dict["selected_nll"].append(
                                prediction["identity_selected_nll"].item()
                            )

                iteration = self.cfg.graph_match.get("iteration", None)
                iteration_suffix = "" if iteration is None else f"_{iteration}"
                direction = self.cfg.train.get("bridge_direction", None)
                # NOTE: direction is set as forward by default
                direction_suffix = "" if iteration is None else f"_{direction}"
                # Epoch (if exists)
                epoch_suffix = ""
                seed_suffix = ""
                # NOTE: If TRUE, WE ARE DOING TEST
                if (
                    ckpt_path := getattr(self.cfg.general, "test_only", None)
                ) is not None:
                    ckpt_path = pathlib.Path(ckpt_path)
                    epoch = ckpt_path.stem.split("=")[-1]
                    epoch_suffix = f"_{epoch}"
                    seed_suffix = f"_seed{self.cfg.general.seed}"

                train_or_test = "train" if not self.cfg.graph_match.test else "test"
                nll_df = pd.DataFrame(nll_dict)
                nll_csv_path = (
                    output_dir
                    / f"{train_or_test}_nll_df{direction_suffix}{iteration_suffix}{epoch_suffix}{seed_suffix}.csv"
                )
                nll_df.to_csv(nll_csv_path)
                self.print(f"Debug] Wrote nll csv to {nll_csv_path}")
                perm_pt_path = (
                    output_dir
                    / f"{train_or_test}_match_perm{direction_suffix}{iteration_suffix}{epoch_suffix}{seed_suffix}.pt"
                )
                torch.save(perm_dict, perm_pt_path)
                self.print(f"Debug] Wrote perm pt to {perm_pt_path}")
                if self.self_recover:
                    randperm_pt_path = (
                        output_dir
                        / f"{train_or_test}_randperm{direction_suffix}{iteration_suffix}{epoch_suffix}{seed_suffix}.pt"
                    )
                    torch.save(randperm_dict, randperm_pt_path)
                    self.print(f"Debug] Wrote randperm pt to {randperm_pt_path}")

                    identity_nll_df = pd.DataFrame(identity_nll_dict)
                    identity_nll_csv_path = (
                        output_dir
                        / f"{train_or_test}_identity_nll_df{direction_suffix}{iteration_suffix}{epoch_suffix}{seed_suffix}.csv"
                    )
                    identity_nll_df.to_csv(identity_nll_csv_path)
                    self.print(
                        f"Debug] Wrote identity nll csv to {identity_nll_csv_path}"
                    )
            else:
                self.print("INSIDE COMPUTE NLL " * 10)
                perm_dict = {}
                nll_dict = defaultdict(list)
                for prediction in predictions:
                    keys, init_nll, selected_nll = (
                        prediction["keys"],
                        prediction["init_nll"],
                        prediction["selected_nll"],
                    )
                    for key, init, selected in zip(keys, init_nll, selected_nll):
                        if key.numel() == 1:
                            k = key.item()
                        else:
                            k = tuple(key.tolist())
                        nll_dict["key"].append(k)
                        nll_dict["init_nll"].append(init.item())
                        nll_dict["selected_nll"].append(selected.item())

                iteration = self.cfg.graph_match.get("iteration", None)
                iteration_suffix = "" if iteration is None else f"_{iteration}"
                direction = self.cfg.train.get("bridge_direction", None)
                # NOTE: direction is set as forward by default
                direction_suffix = "" if iteration is None else f"_{direction}"
                # Epoch (if exists)
                epoch_suffix = ""
                seed_suffix = ""
                # NOTE: If TRUE, WE ARE DOING TEST
                if (
                    ckpt_path := getattr(self.cfg.general, "test_only", None)
                ) is not None:
                    ckpt_path = pathlib.Path(ckpt_path)
                    epoch = ckpt_path.stem.split("=")[-1]
                    epoch_suffix = f"_{epoch}"
                    seed_suffix = f"_seed{self.cfg.general.seed}"

                train_or_test = "train" if not self.cfg.graph_match.test else "test"
                nll_df = pd.DataFrame(nll_dict)
                nll_csv_path = (
                    output_dir
                    / f"recomputed_{train_or_test}_nll_df{direction_suffix}{iteration_suffix}{epoch_suffix}{seed_suffix}.csv"
                )
                nll_df.to_csv(nll_csv_path)
                self.print(f"Debug] Wrote nll csv to {nll_csv_path}")
            return

    def _gather_objects(self, obj):
        if not self.trainer.is_global_zero:
            dist.gather_object(
                obj=obj, object_gather_list=None, dst=0, group=_group.WORLD
            )
            return None
        else:  # global-zero only
            list_gather_obj = [
                None
            ] * self.trainer.world_size  # the container of gathered objects.
            dist.gather_object(
                obj=obj, object_gather_list=list_gather_obj, dst=0, group=_group.WORLD
            )
            return list_gather_obj


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros(len(data.y), 0)
        return data


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


def post_process(cfg, dataset_dir, perm_dir, max_num_nodes: int, train_or_test="train"):
    r"""Post-process the predictions and save the results"""

    print("Debug] Post-processing the predictions")
    ckpt_path = None
    if cfg.graph_match.data_path is None:
        # 1. Load pyg data
        # NOTE: not a data with slices, dictionary of single PairData
        pyg_data_dict = torch.load(dataset_dir / f"{train_or_test}_data.pt")

        # 2. Load perm data
        perm_pt_path = perm_dir / f"processed/{train_or_test}_match_perm.pt"
        assert perm_pt_path.exists(), f"perm_pt_path {perm_pt_path} does not exist"
        perm_dict = torch.load(perm_pt_path)
    else:
        # 1. Load pyg data
        # NOTE: not a data with slices, dictionary of single PairData
        file_path = pathlib.Path(cfg.graph_match.data_path)
        file_list = list(file_path.glob(f"generated_joint_{train_or_test}*.pt"))
        assert len(file_list) == 1, f"file_list {file_list} must have only one file"
        pyg_data_dict = torch.load(file_list[0])

        # 2. Load perm data
        assert (direction := cfg.train.get("bridge_direction", None)) is not None
        assert (iteration := cfg.graph_match.get("iteration", None)) is not None
        direction_suffix = f"_{direction}"
        iteration_suffix = f"_{iteration}"

        epoch_suffix = ""
        seed_suffix = ""
        # NOTE: If TRUE, WE ARE DOING TEST
        if (ckpt_path := getattr(cfg.general, "test_only", None)) is not None:
            ckpt_path = pathlib.Path(ckpt_path)
            epoch = ckpt_path.stem.split("=")[-1]
            epoch_suffix = f"_{epoch}"
            seed_suffix = f"_seed{cfg.general.seed}"

        perm_pt_path = (
            dataset_dir
            / f"processed/{train_or_test}_match_perm{direction_suffix}{iteration_suffix}{epoch_suffix}{seed_suffix}.pt"
        )
        assert perm_pt_path.exists(), f"perm_pt_path {perm_pt_path} does not exist"

        perm_dict = torch.load(perm_pt_path)

    print(f"Debug] len(pyg_dict): {len(pyg_data_dict)}")
    print(f"Debug] len(perm_dict): {len(perm_dict)}")
    assert len(perm_dict) == len(pyg_data_dict), (
        "Number of perm_dict and pyg_data_dict must be the same"
    )

    data_list = []
    pre_transform = RemoveYTransform()

    # 3. Permute the data
    for key in tqdm(perm_dict.keys()):
        # TODO : In previous experiments, there are some unexpected Errors
        # Initial: processed data, Training: generated data
        pyg_data = pyg_data_dict[key]
        perm = perm_dict[key]

        num_nodes = pyg_data.x_T.size(0)
        dense_data_0_p, dense_data_T_p, node_mask_0_p, node_mask_T_p = (
            utils.to_dense_pair(pyg_data, max_num_nodes=num_nodes, target="p")
        )

        X_0, E_0 = dense_data_0_p.X, dense_data_0_p.E
        assert X_0.size(1) == num_nodes, "Num Nodes Error"
        perm = perm[:num_nodes]

        X_0, E_0 = permute(X_0, E_0, perm)
        edge_index_0, edge_attr_0 = dense_to_sparse(torch.argmax(E_0, dim=-1))
        edge_attr_0 = F.one_hot(edge_attr_0, num_classes=E_0.size(-1))

        data = utils.PairData(
            x_0=X_0.squeeze(0) if cfg.graph_match.data_path is None else pyg_data.x_0,
            x_T=pyg_data.x_T,
            x_0_p=X_0.squeeze(0),
            x_T_p=pyg_data.x_T_p,
            edge_index_0=(
                edge_index_0
                if cfg.graph_match.data_path is None
                else pyg_data.edge_index_0
            ),
            edge_index_T=pyg_data.edge_index_T,
            edge_index_0_p=edge_index_0,
            edge_index_T_p=pyg_data.edge_index_T_p,
            edge_attr_0=(
                edge_attr_0
                if cfg.graph_match.data_path is None
                else pyg_data.edge_attr_0
            ),
            edge_attr_T=pyg_data.edge_attr_T,
            edge_attr_0_p=edge_attr_0,
            edge_attr_T_p=pyg_data.edge_attr_T_p,
            num_nodes=pyg_data.num_nodes,
            y=pyg_data.y,
            idx=pyg_data.idx,
        )

        if pre_transform is not None:
            data = pre_transform(data)
        data_list.append(data)

    save_path = osp.join(dataset_dir, "processed", f"{train_or_test}_data.pt")

    print(f"Debug] {save_path}")
    torch.save(_collate(data_list), save_path)
    return


def graph_match(cfg: DictConfig):
    print("Debug] Processing the predictions")
    use_gpu = cfg.graph_match.gpus > 0 and torch.cuda.is_available()

    callbacks = []

    seed = cfg.general.seed
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
    dataset_dir = base_path / cfg.dataset.datadir

    output_dir = dataset_dir

    if not cfg.graph_match.only_post_process:
        dataset_config = cfg["dataset"]
        dataset_name = dataset_config["name"]
        is_unconditional = dataset_config["unconditional"]

        if is_unconditional:
            if dataset_name == "uncond_qm9":
                from ddsbm.datasets.uncond_jointmol_dataset import (
                    JointMolDataModule,
                    JointMolecularinfos,
                )

                datamodule = JointMolDataModule(cfg)
                dataset_infos = JointMolecularinfos(datamodule, cfg)
            elif dataset_name == "uncond_comm20":
                from ddsbm.datasets.uncond_comm20_dataset import (
                    Comm20DataModule,
                    Comm20infos,
                )

                datamodule = Comm20DataModule(cfg)
                dataset_infos = Comm20infos(datamodule, cfg)
            elif dataset_name == "uncond_planar":
                from ddsbm.datasets.uncond_planar_dataset import (
                    PlanarDataModule,
                    Planarinfos,
                )

                datamodule = PlanarDataModule(cfg)
                dataset_infos = Planarinfos(datamodule, cfg)
            else:
                raise ValueError(
                    f"Dataset '{dataset_name}' not supported for unconditional training."
                )
        else:
            from ddsbm.datasets.jointmol_dataset import (
                JointMolDataModule,
                JointMolecularinfos,
            )

            datamodule = JointMolDataModule(cfg)
            dataset_infos = JointMolecularinfos(datamodule, cfg)

        model = GraphMatchModule(cfg=cfg, dataset_dir=dataset_dir)

        trainer = Trainer(
            strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
            accelerator="gpu" if use_gpu else "cpu",
            devices=cfg.graph_match.gpus if use_gpu else 1,
            enable_progress_bar=True,
            callbacks=callbacks,
            logger=[],
        )

        if not cfg.graph_match.test:
            trainer.predict(model=model, dataloaders=datamodule.train_dataloader())
        else:
            trainer.predict(model=model, dataloaders=datamodule.test_dataloader())

        train_or_test = "train" if not cfg.graph_match.test else "test"
        iteration = cfg.graph_match.get("iteration", None)
        iteration_suffix = "" if iteration is None else f"_{iteration}"
        direction = cfg.train.get("bridge_direction", None)
        # NOTE: direction is set as forward by default
        direction_suffix = "" if iteration is None else f"_{direction}"
        epoch_suffix = ""
        seed_suffix = ""
        if (ckpt_path := getattr(cfg.general, "test_only", None)) is not None:
            ckpt_path = pathlib.Path(ckpt_path)
            epoch = ckpt_path.stem.split("=")[-1]
            epoch_suffix = f"_{epoch}"
            seed_suffix = f"_seed{seed}"
        print(f"Debug] ckpt_path: {ckpt_path}")
        cfg_file = (
            dataset_dir
            / f"processed/match_config_{train_or_test}{direction_suffix}{iteration_suffix}{epoch_suffix}{seed_suffix}.yaml"
        )
        with cfg_file.open("w") as f:
            OmegaConf.save(cfg, f)

        # NOTE: do not post-process if test
        if (
            not cfg.graph_match.test
            and cfg.graph_match.post_process
            and model.global_rank == 0
        ):
            post_process(
                cfg, dataset_dir, output_dir, datamodule.max_num_nodes, train_or_test
            )

        trainer.strategy.teardown()
    else:
        dataset_config = cfg["dataset"]
        dataset_name = dataset_config["name"]
        is_unconditional = dataset_config["unconditional"]

        if is_unconditional:
            if dataset_name == "uncond_qm9":
                from ddsbm.datasets.uncond_jointmol_dataset import (
                    JointMolDataModule,
                    JointMolecularinfos,
                )

                datamodule = JointMolDataModule(cfg)
                dataset_infos = JointMolecularinfos(datamodule, cfg)
            elif dataset_name == "uncond_comm20":
                from ddsbm.datasets.uncond_comm20_dataset import (
                    Comm20DataModule,
                    Comm20infos,
                )

                datamodule = Comm20DataModule(cfg)
                dataset_infos = Comm20infos(datamodule, cfg)
            elif dataset_name == "uncond_planar":
                from ddsbm.datasets.uncond_planar_dataset import (
                    PlanarDataModule,
                    Planarinfos,
                )

                datamodule = PlanarDataModule(cfg)
                dataset_infos = Planarinfos(datamodule, cfg)
            else:
                raise ValueError(
                    f"Dataset '{dataset_name}' not supported for unconditional training."
                )
        else:
            from ddsbm.datasets.jointmol_dataset import (
                JointMolDataModule,
                JointMolecularinfos,
            )

            datamodule = JointMolDataModule(cfg)
            dataset_infos = JointMolecularinfos(datamodule, cfg)

        train_or_test = "train" if not cfg.graph_match.test else "test"

        post_process(
            cfg, dataset_dir, output_dir, datamodule.max_num_nodes, train_or_test
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    graph_match(cfg)
