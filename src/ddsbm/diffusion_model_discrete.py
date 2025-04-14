import os
import time
from collections import defaultdict
from pathlib import Path

import pygmtools as pygm
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributed import group as _group
from torch_geometric.utils import dense_to_sparse

from ddsbm import utils
from ddsbm.diffusion import diffusion_utils
from ddsbm.diffusion.noise_schedule import (
    DiscreteUniformTransition,
    MarginalUniformTransition,
    PredefinedNoiseScheduleDiscrete,
    SymmetricNoiseScheduleDiscrete,
)
from ddsbm.metrics.abstract_metrics import NLL, SumExceptBatchKL
from ddsbm.metrics.train_metrics import TrainLossDiscreteKL
from ddsbm.models.transformer_model import GraphTransformer

pygm.set_backend("pytorch")


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
    ):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.dataset_name = cfg.dataset.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.max_num_nodes = dataset_infos.max_num_nodes
        print(f"Debug] Max num nodes: {self.max_num_nodes}")

        self.Xdim = input_dims["X"]
        self.Edim = input_dims["E"]
        self.ydim = input_dims["y"]
        self.Xdim_output = output_dims["X"]
        self.Edim_output = output_dims["E"]
        self.ydim_output = output_dims["y"]
        print("Debug] Dimension Info)", end=" ")
        print(f"Xdim: {self.Xdim}, Edim: {self.Edim}, ydim: {self.ydim}")

        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        # NOTE : For dummy node index
        # self.dummy_index = self.dataset_info.atom_decoder.index("X")

        self.train_kl_loss = TrainLossDiscreteKL(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        if cfg.model.symmetric_noise_schedule:
            print("Debug] Using symmetric noise schedule")
            self.noise_schedule = SymmetricNoiseScheduleDiscrete(
                cfg.model.diffusion_noise_schedule,
                timesteps=cfg.model.diffusion_steps,
                min_alpha=cfg.model.min_alpha,
            )
        else:
            print("Debug] Using asymmetric noise schedule")
            self.noise_schedule = PredefinedNoiseScheduleDiscrete(
                cfg.model.diffusion_noise_schedule, timesteps=cfg.model.diffusion_steps
            )

        if cfg.model.transition == "uniform":
            # NOTE: Same as num node types, num edge typs (including dummy)
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output,
            )
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif cfg.model.transition == "marginal":
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print("Debug] Marginal distribution of the classes:", end="\n")
            print(f"{x_marginals} for nodes,", end="\n")
            print(f"{e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.ydim_output,
            )
            self.limit_dist = utils.PlaceHolder(
                X=x_marginals,
                E=e_marginals,
                y=torch.ones(self.ydim_output) / self.ydim_output,
            )

        else:
            raise ValueError(
                "transition should be selected between uniform and marginal"
            )

        self.loss_type = cfg.train.loss_type
        assert self.loss_type == "KL", "Only KL loss is supported"

        self.bridge_direction = cfg.train.bridge_direction
        assert self.bridge_direction in [
            "forward",
            "backward",
        ], "Invalid bridge direction"

        self.save_hyperparameters(
            ignore=["train_metrics", "sampling_metrics", "dataset_infos"]
        )
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.gen_data_dicts = []
        self.gen_mol_dicts = []
        self.predict_step_outputs = {}

    def to_sparse(self, X, E, node_mask):
        """
        Convert dense data with empty values and masks into sparse,
        values and corresponding indice.

        """

        # Make adjacency matrix (b, n, n, d) to (b, n, n)
        E = torch.argmax(E, dim=-1)

        xs = []
        edge_indexs = []
        edge_attrs = []
        for data in zip(X, E, node_mask):
            x, e, mask = data
            x = x[mask]
            edge_index, edge_attr = dense_to_sparse(e)
            edge_attr = F.one_hot(edge_attr, num_classes=self.Edim_output)
            xs.append(x)
            edge_indexs.append(edge_index)
            edge_attrs.append(edge_attr)

        return xs, edge_indexs, edge_attrs

    def random_permute(self, dense_data, node_mask):
        """Randomly permute the nodes of the graph."""
        x = dense_data.X
        e = dense_data.E
        num_nodes = x.size(1)
        perm = torch.randperm(num_nodes)
        perm_x = x[:, perm, :]
        perm_e = e[:, perm, :, :][:, :, perm, :]
        dense_data.X = perm_x
        dense_data.E = perm_e
        perm_node_mask = node_mask[:, perm]
        return dense_data, perm_node_mask

    def bridge_forward(self, noisy_data, pred):
        """Forward bridge process q(x_t+1|x_t, x_T)
        Args:
            noisy_data: Dict containing the noisy data
            pred: Dict containing the predicted distribution
        Returns:
            kl_loss: KL divergence between the true and predicted distributions
        """
        node_mask = noisy_data["node_mask"]

        t_float = noisy_data["t"]
        u_float = noisy_data["u"]
        T_float = noisy_data["T"]

        beta_u = self.noise_schedule(t_normalized=u_float)  # (bs, 1)

        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)
        alpha_u_bar = self.noise_schedule.get_alpha_bar(t_normalized=u_float)  # (bs, 1)
        alpha_T_bar = self.noise_schedule.get_alpha_bar(t_normalized=T_float)  # (bs, 1)

        Qu = self.transition_model.get_Qt(beta_u, self.device)  # (bs, n ,d_t, d_t+1)

        QuTb = self.transition_model.get_Qst_bar(
            alpha_t_bar, alpha_T_bar, device=self.device
        )

        QvTb = self.transition_model.get_Qst_bar(
            alpha_u_bar, alpha_T_bar, device=self.device
        )

        X_t, E_t, y_t = noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        X_T, E_T, y_T = noisy_data["X_T"], noisy_data["E_T"], noisy_data["y_T"]
        bs, n, d0 = noisy_data["X_0"].shape

        prob_true_X = diffusion_utils.compute_forward_distribution(
            X_t, X_T, Qu.X, QuTb.X, QvTb.X
        )
        prob_true_E = diffusion_utils.compute_forward_distribution(
            E_t, E_T, Qu.E, QuTb.E, QvTb.E
        )
        prob_true_E = prob_true_E.reshape((bs, n, n, -1))

        pred_probs_X = F.softmax(pred.X, dim=-1)  # X_T
        pred_probs_E = F.softmax(pred.E, dim=-1)  # E_T
        pred_probs_y = F.softmax(pred.y, dim=-1)  # y_T

        prob_pred_X = diffusion_utils.compute_forward_distribution(
            X_t, pred_probs_X, Qu.X, QuTb.X, QvTb.X
        )
        prob_pred_E = diffusion_utils.compute_forward_distribution(
            E_t, pred_probs_E, Qu.E, QuTb.E, QvTb.E
        )
        prob_pred_E = prob_pred_E.reshape((bs, n, n, -1))

        prob_true_X, prob_true_E, prob_pred_X, prob_pred_E = (
            diffusion_utils.mask_distributions(
                true_X=prob_true_X,
                true_E=prob_true_E,
                pred_X=prob_pred_X,
                pred_E=prob_pred_E,
                node_mask=node_mask,
            )
        )

        kl_loss = self.train_kl_loss(prob_true_X, prob_pred_X, prob_true_E, prob_pred_E)

        return kl_loss

    def bridge_backward(self, noisy_data, pred):
        """Backward bridge process q(x_t-1|x_t, x_0)
        Args:
            noisy_data: Dict containing the noisy data
            pred: Dict containing the predicted distribution
        Returns:
            kl_loss: KL divergence between the true and predicted distributions
        """
        node_mask = noisy_data["node_mask"]
        t_float = noisy_data["t"]
        s_float = noisy_data["s"]

        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)

        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qt = self.transition_model.get_Qt(beta_t, self.device)  # (bs, n ,d_t, d_t+1)

        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=self.device)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)

        X_t, E_t, y_t = noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        X_0, E_0, y_0 = noisy_data["X_0"], noisy_data["E_0"], noisy_data["y_0"]
        bs, n, d0 = noisy_data["X_0"].shape

        prob_true_X = diffusion_utils.compute_backward_distribution(
            X_t, X_0, Qt.X, Qsb.X, Qtb.X
        )
        prob_true_E = diffusion_utils.compute_backward_distribution(
            E_t, E_0, Qt.E, Qsb.E, Qtb.E
        )
        prob_true_E = prob_true_E.reshape((bs, n, n, -1))

        pred_probs_X = F.softmax(pred.X, dim=-1)  # X_0
        pred_probs_E = F.softmax(pred.E, dim=-1)  # E_0
        pred_probs_y = F.softmax(pred.y, dim=-1)  # y_0

        prob_pred_X = diffusion_utils.compute_backward_distribution(
            X_t, pred_probs_X, Qt.X, Qsb.X, Qtb.X
        )
        prob_pred_E = diffusion_utils.compute_backward_distribution(
            E_t, pred_probs_E, Qt.E, Qsb.E, Qtb.E
        )
        prob_pred_E = prob_pred_E.reshape((bs, n, n, -1))

        prob_true_X, prob_true_E, prob_pred_X, prob_pred_E = (
            diffusion_utils.mask_distributions(
                true_X=prob_true_X,
                true_E=prob_true_E,
                pred_X=prob_pred_X,
                pred_E=prob_pred_E,
                node_mask=node_mask,
            )
        )

        kl_loss = self.train_kl_loss(prob_true_X, prob_pred_X, prob_true_E, prob_pred_E)

        return kl_loss

    def training_step(self, data, i):
        st = time.time()
        if data.edge_index_0.numel() == 0 or data.edge_index_T.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        y = data.y
        # Load coupled data
        dense_data_0, dense_data_T, node_mask_0, node_mask_T = utils.to_dense_pair(
            data=data, max_num_nodes=self.max_num_nodes, target="p"
        )
        assert (node_mask_0 == node_mask_T).all(), "Node masks are not equal"
        node_mask = node_mask_0

        dense_data_0 = dense_data_0.mask(node_mask)
        dense_data_T = dense_data_T.mask(node_mask)

        X_0, E_0 = dense_data_0.X, dense_data_0.E
        X_T, E_T = dense_data_T.X, dense_data_T.E

        # NOTE : Apply Noise to the data
        noisy_data = self.apply_bridge_noise(
            X_0, X_T, E_0, E_T, y, node_mask, self.bridge_direction
        )

        extra_data = self.compute_extra_pair_data(noisy_data)

        # NOTE : Predict target graph from noised graph
        pred = self.forward(noisy_data, extra_data, node_mask)

        if self.bridge_direction == "forward":
            loss = self.bridge_forward(noisy_data, pred)
        else:
            loss = self.bridge_backward(noisy_data, pred)

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Debug] Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)
        self.print(
            f"Debug] Training Dataset Size : {len(self.trainer.datamodule.train_dataset)}"
        )
        self.print(
            f"Debug] Validation Dataset Size : {len(self.trainer.datamodule.val_dataset)}"
        )

    def on_train_epoch_start(self) -> None:
        self.print("Debug] Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_kl_loss.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_kl_loss.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: X_KL: {to_log['train_epoch/X_KL']:.3f} --"
            f" -- E_KL: {to_log['train_epoch/E_KL']:.3f} --"
            f" -- {time.time() - self.start_epoch_time:.1f}s "
        )
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        if data.edge_index_0.numel() == 0 or data.edge_index_T.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        y = data.y
        # Load coupled data
        dense_data_0, dense_data_T, node_mask_0, node_mask_T = utils.to_dense_pair(
            data=data, max_num_nodes=self.max_num_nodes, target="p"
        )
        assert (node_mask_0 == node_mask_T).all(), "Node masks are not equal"
        node_mask = node_mask_0

        dense_data_0 = dense_data_0.mask(node_mask)
        dense_data_T = dense_data_T.mask(node_mask)

        X_0, E_0 = dense_data_0.X, dense_data_0.E
        X_T, E_T = dense_data_T.X, dense_data_T.E

        noisy_data = self.apply_bridge_noise(
            X_0, X_T, E_0, E_T, y, node_mask, self.bridge_direction
        )

        extra_data = self.compute_extra_pair_data(noisy_data)

        pred = self.forward(noisy_data, extra_data, node_mask)

        nll = self.compute_val_loss(pred, noisy_data, X_0, E_0, X_T, E_T, y, node_mask)
        return {"loss": nll}

    def on_validation_epoch_end(self) -> None:
        # NOTE: This function also runs in N gpus separately
        metrics = [
            self.val_nll.compute(),
            self.val_X_kl.compute() * self.T,
            self.val_E_kl.compute() * self.T,
        ]
        if wandb.run:
            wandb.log(
                {
                    "val/epoch_NLL": metrics[0],
                    "val/X_kl": metrics[1],
                    "val/E_kl": metrics[2],
                },
                commit=False,
            )

        self.print(
            f"Epoch {self.current_epoch}: Val NLL {metrics[0]:.2f} -- Val Node type KL {metrics[1]:.2f} -- ",
            f"Val Edge type KL: {metrics[2]:.2f}",
        )

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print(
            "Val loss: %.4f \t Best val loss:  %.4f\n" % (val_nll, self.best_val_nll)
        )

        # FIXME: With 2 GPUs, ran twice?
        # total validation data: 3000, batch size: 150, but with 2 GPUs,
        # Sampling indent becomes higher than 1500
        # TODO: use sample_batch_size in the val_dataloader??
        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            self.print("Debug] Starting validation sampling...")
            start = time.time()

            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []
            ident = 0

            for i, data in enumerate(self.trainer.datamodule.val_dataloader()):
                if ident >= samples_left_to_generate:
                    break
                self.print(f"Debug] Sampling {ident} th")
                data = data.to(self.device)

                # Load coupled data
                dense_data_0, dense_data_T, node_mask_0, node_mask_T = (
                    utils.to_dense_pair(
                        data=data, max_num_nodes=self.max_num_nodes, target="p"
                    )
                )
                assert (node_mask_0 == node_mask_T).all(), "Node masks are not equal"
                node_mask = node_mask_0
                y = data.y

                dense_data_0 = dense_data_0.mask(node_mask)
                dense_data_T = dense_data_T.mask(node_mask)

                X_0, E_0 = (dense_data_0.X, dense_data_0.E)
                X_T, E_T = (dense_data_T.X, dense_data_T.E)

                bs = X_0.size(0)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)

                if self.bridge_direction == "forward":
                    samples.extend(
                        self.sample_forward_bridge_batch(
                            X_0=X_0,
                            E_0=E_0,
                            y=y,
                            node_mask=node_mask_0,
                            batch_id=ident,
                            batch_size=bs,
                            num_nodes=None,
                            save_final=to_save,
                            keep_chain=chains_save,
                            number_chain_steps=self.number_chain_steps,
                        )[0]
                    )
                else:
                    samples.extend(
                        self.sample_backward_bridge_batch(
                            X_T=X_T,
                            E_T=E_T,
                            y=data.y,
                            node_mask=node_mask_T,
                            batch_id=ident,
                            batch_size=bs,
                            num_nodes=None,
                            save_final=to_save,
                            keep_chain=chains_save,
                            number_chain_steps=self.number_chain_steps,
                        )[0]
                    )

                ident += bs

                samples_left_to_save -= to_save
                chains_left_to_save -= chains_save

            self.print("Computing sampling metrics...")
            self.sampling_metrics.forward(
                samples,
                self.name,
                self.current_epoch,
                val_counter=-1,
                test=False,
                local_rank=self.local_rank,  # FIXME: should handle local rank in here?
            )
            self.print(f"Done. Sampling took {time.time() - start:.2f} seconds\n")
            self.print("Validation epoch end ends...")
        torch.cuda.empty_cache()

    def on_test_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.sampling_metrics.reset()

    def test_step(self, data, i):
        nll = torch.tensor([0.0])
        return {"loss": nll}

    def on_test_epoch_end(self) -> None:
        """Measure likelihood on a test set and compute stability metrics."""
        self.print(f"Epoch {self.current_epoch}: No Validation")

        self.print(f"Debug] Starting test sampling...")
        start = time.time()

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save
        # samples_left_to_save = 0
        # chains_left_to_save = 0
        print(f"Debug] samples_left_to_save : {samples_left_to_save}")
        print(f"Debug] chains_left_to_save : {chains_left_to_save}")

        samples = []
        ident = 0

        for i, data in enumerate(self.trainer.datamodule.test_dataloader()):
            self.print(f"Debug] Sampling {ident} th")
            data = data.to(self.device)

            # Load coupled data
            dense_data_0, dense_data_T, node_mask_0, node_mask_T = utils.to_dense_pair(
                data=data, max_num_nodes=self.max_num_nodes, target="p"
            )
            assert (node_mask_0 == node_mask_T).all(), "Node masks are not equal"
            node_mask = node_mask_0
            y = data.y

            dense_data_0 = dense_data_0.mask(node_mask_0)
            dense_data_T = dense_data_T.mask(node_mask_T)

            # X_0, E_0 = (
            #     dense_data_0.X,
            #     dense_data_0.E,
            # )  # (bs, n, d0), (bs, n, n, d0)
            # X_T, E_T = (
            #     dense_data_T.X,
            #     dense_data_T.E,
            # )  # (bs, n, dT), (bs, n, n, dT)

            # NOTE : Sample N and Sample prior
            bsz = dense_data_0.X.size(0)
            n_nodes = self.node_dist.sample_n(bsz, device=dense_data_0.X.device)
            # n_max = torch.max(n_nodes).item()
            n_max = self.max_num_nodes
            arange = (
                torch.arange(n_max, device=self.device).unsqueeze(0).expand(bsz, -1)
            )
            node_mask_0 = arange < n_nodes.unsqueeze(1)
            z_T = diffusion_utils.sample_discrete_feature_noise(
                limit_dist=self.limit_dist, node_mask=node_mask_0
            )  # NOTE : True Prior Distribution

            X_0, E_0 = z_T.X, z_T.E
            X_T, E_T = dense_data_T.X, dense_data_T.E

            bs = X_0.size(0)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)

            if self.bridge_direction == "forward":
                samples.extend(
                    self.sample_forward_bridge_batch(
                        X_0=X_0,
                        E_0=E_0,
                        y=y,
                        node_mask=node_mask_0,
                        batch_id=ident,
                        batch_size=bs,
                        num_nodes=None,
                        save_final=to_save,
                        keep_chain=chains_save,
                        number_chain_steps=self.number_chain_steps,
                    )[0]
                )
            else:
                samples.extend(
                    self.sample_backward_bridge_batch(
                        X_T=X_T,
                        E_T=E_T,
                        y=y,
                        node_mask=node_mask_T,
                        batch_id=ident,
                        batch_size=bs,
                        num_nodes=None,
                        save_final=to_save,
                        keep_chain=chains_save,
                        number_chain_steps=self.number_chain_steps,
                    )[0]
                )

            ident += bs

            samples_left_to_save -= to_save
            chains_left_to_save -= chains_save

            # MSEOK: FOR FASTTER SAMPLING, SHOULD REMOVE THIS!
            if ident >= samples_left_to_generate:
                break

        self.print("Saving the generated graphs")
        filename = f"generated_samples1.txt"
        for i in range(2, 10):
            if os.path.exists(filename):
                filename = f"generated_samples{i}.txt"
            else:
                break

        self.print(f"Saving samples to {filename}")

        with open(filename, "w") as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")

        self.print("Computing sampling metrics...")
        self.sampling_metrics.forward(
            samples,
            self.name,
            self.current_epoch,
            val_counter=-1,
            test=True,
            local_rank=self.local_rank,
        )
        self.print(f"Done. Sampling took {time.time() - start:.2f} seconds\n")
        self.print("Validation epoch end ends...")

    def on_predict_epoch_start(self) -> None:
        self.print("Starting Sampling...")

    def predict_step(self, data, i):
        self.print(f"Sampling {i}...")

        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        # Load coupled data
        # NOTE: IN SAMPLING WE SHOULD USE 0,
        # using original data to generate new data (either forward or backward)
        dense_data_0, dense_data_T, node_mask_0, node_mask_T = utils.to_dense_pair(
            data=data, max_num_nodes=self.max_num_nodes, target="0"
        )
        assert (node_mask_0 == node_mask_T).all(), "Node masks are not equal"
        node_mask = node_mask_0
        y = data.y
        data_list = data.to_data_list()

        dense_data_0 = dense_data_0.mask(node_mask)
        dense_data_T = dense_data_T.mask(node_mask)

        X_0, E_0 = dense_data_0.X, dense_data_0.E
        X_T, E_T = dense_data_T.X, dense_data_T.E

        bs = X_0.size(0)
        to_save = min(samples_left_to_save, bs)
        chains_save = min(chains_left_to_save, bs)

        data_dict = {}
        output = defaultdict(list)
        if self.bridge_direction == "forward":
            generated, gen_X, gen_E = self.sample_forward_bridge_batch(
                X_0=X_0,
                E_0=E_0,
                y=y,
                node_mask=node_mask,
                batch_id=i,
                batch_size=bs,
                num_nodes=None,
                save_final=to_save,
                keep_chain=chains_save,
                number_chain_steps=self.number_chain_steps,
            )

            gen_x, gen_edge_index, gen_edge_attr = self.to_sparse(
                gen_X, gen_E, node_mask
            )

            for idx, data in enumerate(data_list):
                # NOTE: These data do not have padding and masks
                new_data = utils.PairData(
                    x_0=data.x_0,
                    x_T=data.x_T,
                    x_0_p=data.x_0,
                    x_T_p=gen_x[idx],
                    edge_index_0=data.edge_index_0,
                    edge_index_T=data.edge_index_T,
                    edge_index_0_p=data.edge_index_0,
                    edge_index_T_p=gen_edge_index[idx],
                    edge_attr_0=data.edge_attr_0,
                    edge_attr_T=data.edge_attr_T,
                    edge_attr_0_p=data.edge_attr_0,
                    edge_attr_T_p=gen_edge_attr[idx],
                    y=data.y,
                    idx=data.idx,
                )

                data_idx = data.idx.item()
                data_dict[data_idx] = new_data

        else:
            generated, gen_X, gen_E = self.sample_backward_bridge_batch(
                X_T=X_T,
                E_T=E_T,
                y=data.y,
                node_mask=node_mask,
                batch_id=i,
                batch_size=bs,
                num_nodes=None,
                save_final=to_save,
                keep_chain=chains_save,
                number_chain_steps=self.number_chain_steps,
            )

            gen_x, gen_edge_index, gen_edge_attr = self.to_sparse(
                gen_X, gen_E, node_mask
            )

            for idx, data in enumerate(data_list):
                new_data = utils.PairData(
                    x_0=data.x_0,
                    x_T=data.x_T,
                    x_0_p=gen_x[idx],
                    x_T_p=data.x_T,
                    edge_index_0=data.edge_index_0,
                    edge_index_T=data.edge_index_T,
                    edge_index_0_p=gen_edge_index[idx],
                    edge_index_T_p=data.edge_index_T,
                    edge_attr_0=data.edge_attr_0,
                    edge_attr_T=data.edge_attr_T,
                    edge_attr_0_p=gen_edge_attr[idx],
                    edge_attr_T_p=data.edge_attr_T,
                    y=data.y,
                    idx=data.idx,
                )

                data_idx = data.idx.item()
                data_dict[data_idx] = new_data

        self.predict_step_outputs.update(data_dict)
        return {"graph_data": data_dict}

    def on_predict_epoch_end(self) -> None:
        list_gather_step_outputs = self._gather_objects(obj=self.predict_step_outputs)
        if not self.trainer.is_global_zero:
            return
        else:  # global zero only
            assert len(list_gather_step_outputs) == self.trainer.world_size
            data_dict = {}
            for outputs in list_gather_step_outputs:
                data_dict.update(outputs)

            # sort dictionary by key
            data_dict = dict(sorted(data_dict.items()))

            output_dir = Path(self.cfg.graph_match.data_path)
            # trainer test or train
            if getattr(self.trainer, "datamodule", None) is not None:
                train_or_test = "train"
            else:
                train_or_test = "test"
            file = f"generated_joint_{train_or_test}_seed{self.cfg.general.seed}_nfe{self.T}.pt"

            self.print(f"Debug] Writing predictions to {output_dir / file}")
            torch.save(data_dict, output_dir / file)
            self.print(f"Debug] Writing predictions to {output_dir / file} Done.")

            # NOTE: In both Training and Test, to do graph matching, we should merge data
            if train_or_test == "train":
                save_path = (
                    self.trainer.datamodule.root_path / "processed/train_data.pt"
                )
            else:
                save_path = (
                    self.trainer.predict_dataloaders.dataset.root
                    / "processed/test_data.pt"
                )

            assert save_path.exists(), f"File not found: {save_path}"
            self.print(
                f"Debug] Merge predictions in {output_dir / file} to {save_path}"
            )
            utils.merge(output_dir / file, save_path)
            self.print(
                f"Debug] Merge predictions in {output_dir / file} to {save_path} Done."
            )
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

    def compute_forward_Lt(self, X_T, E_T, y, pred, noisy_data, node_mask, test):
        """Compute the forward KL divergence between the true and predicted distributions
        Args:
            X_T (torch.Tensor) : True node features at time T
            E_T (torch.Tensor) : True edge features at time T
            y (torch.Tensor) : True target labels
            pred (Dict) : Predicted distribution
            noisy_data (Dict) : Noisy data
            node_mask (torch.Tensor) : Node mask
            test (bool) : Whether the model is in test mode
        Returns:
            kl_loss (torch.Tensor) : Forward KL divergence
        """
        bs, n, _ = X_T.shape
        t_float = noisy_data["t"]
        u_float = noisy_data["u"]
        T_float = noisy_data["T"]

        beta_u = self.noise_schedule(t_normalized=u_float)  # (bs, 1)

        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)
        alpha_u_bar = self.noise_schedule.get_alpha_bar(t_normalized=u_float)  # (bs, 1)
        alpha_T_bar = self.noise_schedule.get_alpha_bar(t_normalized=T_float)  # (bs, 1)

        Qu = self.transition_model.get_Qt(beta_u, self.device)  # (bs, n ,d_t, d_t+1)

        QuTb = self.transition_model.get_Qst_bar(
            alpha_t_bar, alpha_T_bar, device=self.device
        )

        QvTb = self.transition_model.get_Qst_bar(
            alpha_u_bar, alpha_T_bar, device=self.device
        )

        X_t, E_t, y_t = noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]

        prob_true_X = diffusion_utils.compute_forward_distribution(
            X_t, X_T, Qu.X, QuTb.X, QvTb.X
        )
        prob_true_E = diffusion_utils.compute_forward_distribution(
            E_t, E_T, Qu.E, QuTb.E, QvTb.E
        )
        prob_true_E = prob_true_E.reshape((bs, n, n, -1))

        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        prob_pred_X = diffusion_utils.compute_forward_distribution(
            X_t, pred_probs_X, Qu.X, QuTb.X, QvTb.X
        )
        prob_pred_E = diffusion_utils.compute_forward_distribution(
            E_t, pred_probs_E, Qu.E, QuTb.E, QvTb.E
        )
        prob_pred_E = prob_pred_E.reshape((bs, n, n, -1))

        prob_true_X, prob_true_E, prob_pred_X, prob_pred_E = (
            diffusion_utils.mask_distributions(
                true_X=prob_true_X,
                true_E=prob_true_E,
                pred_X=prob_pred_X,
                pred_E=prob_pred_E,
                node_mask=node_mask,
            )
        )

        kl_x = (self.test_X_kl if test else self.val_X_kl)(
            prob_true_X, torch.log(prob_pred_X)
        )
        kl_e = (self.test_E_kl if test else self.val_E_kl)(
            prob_true_E, torch.log(prob_pred_E)
        )
        return self.T * (kl_x + kl_e)

    def compute_backward_Lt(self, X_0, E_0, y, pred, noisy_data, node_mask, test):
        """Compute the backward KL divergence between the true and predicted distributions
        Args:
            X_0 (torch.Tensor) : True node features at time 0
            E_0 (torch.Tensor) : True edge features at time 0
            y (torch.Tensor) : True target labels
            pred (Dict) : Predicted distribution
            noisy_data (Dict) : Noisy data
            node_mask (torch.Tensor) : Node mask
            test (bool) : Whether the model is in test mode
        Returns:
            kl_loss (torch.Tensor) : Backward KL divergence
        """
        t_float = noisy_data["t"]
        s_float = noisy_data["s"]

        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qt = self.transition_model.get_Qt(beta_t, self.device)  # (bs, n ,d_t, d_t+1)

        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=self.device)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)

        X_t, E_t, y_t = noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]

        bs, n, d0 = noisy_data["X_0"].shape

        prob_true_X = diffusion_utils.compute_backward_distribution(
            X_t, X_0, Qt.X, Qsb.X, Qtb.X
        )
        prob_true_E = diffusion_utils.compute_backward_distribution(
            E_t, E_0, Qt.E, Qsb.E, Qtb.E
        )
        prob_true_E = prob_true_E.reshape((bs, n, n, -1))

        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        prob_pred_X = diffusion_utils.compute_backward_distribution(
            X_t, pred_probs_X, Qt.X, Qsb.X, Qtb.X
        )
        prob_pred_E = diffusion_utils.compute_backward_distribution(
            E_t, pred_probs_E, Qt.E, Qsb.E, Qtb.E
        )
        prob_pred_E = prob_pred_E.reshape((bs, n, n, -1))

        prob_true_X, prob_true_E, prob_pred_X, prob_pred_E = (
            diffusion_utils.mask_distributions(
                true_X=prob_true_X,
                true_E=prob_true_E,
                pred_X=prob_pred_X,
                pred_E=prob_pred_E,
                node_mask=node_mask,
            )
        )

        kl_x = (self.test_X_kl if test else self.val_X_kl)(
            prob_true_X, torch.log(prob_pred_X)
        )
        kl_e = (self.test_E_kl if test else self.val_E_kl)(
            prob_true_E, torch.log(prob_pred_E)
        )
        return self.T * (kl_x + kl_e)

    def apply_bridge_noise(self, X_0, X_T, E_0, E_T, y, node_mask, bridge_direction):
        """Apply bridge transition matrices to the data
        Args:
            X_0 (torch.Tensor): Node features at time 0 (bs, n, dx)
            X_T (torch.Tensor): Node features at time T (bs, n, dx)
            E_0 (torch.Tensor): Edge features at time 0 (bs, n, n, de)
            E_T (torch.Tensor): Edge features at time T (bs, n, n, de)
            y (torch.Tensor): Target (bs, dy)
            node_mask (torch.Tensor): Node mask (bs, n)
            bridge_direction (str): Direction of the bridge transition
        """
        bs, n, dxs = X_0.shape

        if bridge_direction == "forward":
            # t_int should be in [1, T-2]
            t_int = torch.randint(
                1, self.T - 1, size=(X_0.size(0), 1), device=X_0.device
            ).float()  # (bs, 1)
        else:
            # t_int should be in [2, T-1]
            t_int = torch.randint(
                2, self.T, size=(X_0.size(0), 1), device=X_0.device
            ).float()  # (bs, 1)

        s_int = t_int - 1
        u_int = t_int + 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        u_float = u_int / self.T
        T_float = torch.ones(t_float.size()).float()

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)
        alpha_T_bar = self.noise_schedule.get_alpha_bar(t_normalized=T_float)

        # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)
        QTb = self.transition_model.get_Qt_bar(alpha_T_bar, device=self.device)
        QuTb = self.transition_model.get_Qst_bar(
            alpha_t_bar, alpha_T_bar, device=self.device
        )

        assert (abs(Qtb.X.sum(dim=2) - 1.0) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.0) < 1e-4).all()
        assert (abs(QuTb.X.sum(dim=2) - 1.0) < 1e-4).all(), QuTb.X.sum(dim=2) - 1
        assert (abs(QuTb.E.sum(dim=2) - 1.0) < 1e-4).all()
        assert (abs(QTb.X.sum(dim=2) - 1.0) < 1e-4).all(), QTb.X.sum(dim=2) - 1
        assert (abs(QTb.E.sum(dim=2) - 1.0) < 1e-4).all()

        mask = node_mask.unsqueeze(-1)

        probX = diffusion_utils.compute_marginal_distribution(
            X_0, X_T, Qtb.X, QTb.X, QuTb.X
        )
        probE = diffusion_utils.compute_marginal_distribution(
            E_0, E_T, Qtb.E, QTb.E, QuTb.E
        )
        probE = probE.reshape(bs, n, n, -1)

        sampled_t = diffusion_utils.sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask
        )

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X_0.shape == X_t.shape) and (E_0.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            "s_int": s_int,
            "s": s_float,
            "t_int": t_int,
            "t": t_float,
            "u_int": u_int,
            "u": u_float,
            "T": T_float,
            "beta_t": beta_t,
            "X_0": X_0,
            "E_0": E_0,
            "y_0": z_t.y,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "X_T": X_T,
            "E_T": E_T,
            "y_T": z_t.y,
            "node_mask": node_mask,
        }

        return noisy_data

    def compute_val_loss(
        self, pred, noisy_data, X_0, E_0, X_T, E_T, y, node_mask, test=False
    ):
        """Computes an estimator for the variational lower bound.
        pred: (batch_size, n, total_features)
        noisy_data: dict
        X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
        node_mask : (bs, n)
        Output: nll (size 1)
        """
        if self.bridge_direction == "forward":
            loss_all_t = self.compute_forward_Lt(
                X_T, E_T, y, pred, noisy_data, node_mask, test
            )
        else:
            loss_all_t = self.compute_backward_Lt(
                X_0, E_0, y, pred, noisy_data, node_mask, test
            )
        nlls = loss_all_t.unsqueeze(-1)
        assert len(nlls.shape) == 1, f"{nlls.shape} has more than only batch dim."

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)  # Average over the batch

        if wandb.run:
            wandb.log(
                {
                    "Estimator loss terms": loss_all_t.mean(),
                    "batch_test_nll" if test else "val_nll": nll,
                },
                commit=False,
            )
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_forward_bridge_batch(
        self,
        X_0: torch.tensor,
        E_0: torch.tensor,
        y: torch.tensor,
        node_mask: torch.tensor,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        num_nodes=None,
    ):
        """
        :param X_T: torch.tensor
        :param E_T: torch.tensor
        :param node_mask: torch.tensor
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        z_0 = utils.PlaceHolder(X=X_0, E=E_0, y=y).type_as(X_0).mask(node_mask)

        # Start sampling from input X_0, E_0
        X, E, y = z_0.X, z_0.E, z_0.y  # (bs, n, d0), (bs, n, n, d0)

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T

        chain_X_size = torch.Size(
            (number_chain_steps, keep_chain, X.size(1))
        )  # e.g. (50, 30, n)
        chain_E_size = torch.Size(
            (number_chain_steps, keep_chain, E.size(1), E.size(2))
        )  # e.g. (50, 30, n, n)

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Make n_nodes tensor using X_T
        n_nodes = torch.sum(node_mask, dim=1)

        # Iteratively sample p(z_u | z_t) for t = 0, ..., T-1, with u = t + 1.
        # NOTE : q(x_T | x_T-1, x_T) = 1
        for t_int in range(0, self.T - 1):
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            u_array = t_array + 1
            t_norm = t_array / self.T
            u_norm = u_array / self.T

            # Sample z_s
            sampled_u, discrete_sampled_u = self.sample_bridge_p_zu_given_zt(
                t_norm, u_norm, X, E, y, node_mask
            )
            X, E, y = sampled_u.X, sampled_u.E, sampled_u.y

            write_index = (t_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_u.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_u.E[:keep_chain]

        # Sample
        gen_X, gen_E = sampled_u.X, sampled_u.E
        sampled_u = sampled_u.mask(node_mask, collapse=True)
        X, E, y = sampled_u.X, sampled_u.E, sampled_u.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            # Overwrite last frame with the resulting X, E
            chain_X[-1] = final_X_chain
            chain_E[-1] = final_E_chain

            z_0_discrete = (
                utils.PlaceHolder(X=X_0, E=E_0, y=y)
                .type_as(X_0)
                .mask(node_mask, collapse=True)
            )

            chain_X[0] = z_0_discrete.X[:keep_chain]
            chain_E[0] = z_0_discrete.E[:keep_chain]

            # Repeat last frame to see final sample better
            # BUG: should redo chain visualization
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        try:
            if self.visualization_tools is not None:
                self.print("Visualizing chains...")
                current_path = os.getcwd()
                num_molecules = chain_X.size(1)  # number of molecules
                for i in range(num_molecules):
                    result_path = os.path.join(
                        current_path,
                        f"chains/{self.cfg.general.name}/"
                        f"epoch{self.current_epoch}/"
                        f"chains/molecule_{batch_id + i}",
                    )
                    if not os.path.exists(result_path):
                        os.makedirs(result_path, exist_ok=True)
                        _ = self.visualization_tools.visualize_chain(
                            result_path,
                            chain_X[:, i, :].numpy(),
                            chain_E[:, i, :].numpy(),
                        )
                    self.print("\r{}/{} complete".format(i + 1, num_molecules))
                self.print("\nVisualizing molecules...")

                # Visualize the final molecules
                current_path = os.getcwd()
                result_path = os.path.join(
                    current_path,
                    f"graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/",
                )
                self.visualization_tools.visualize(
                    result_path, molecule_list, save_final
                )
                self.print("Done.")
        except Exception as e:
            print(f"Debug] rank: {self.local_rank} Error in visualization : {e}")

        return molecule_list, gen_X, gen_E

    @torch.no_grad()
    def sample_backward_bridge_batch(
        self,
        X_T: torch.tensor,
        E_T: torch.tensor,
        y: torch.tensor,
        node_mask: torch.tensor,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        num_nodes=None,
    ):
        """
        :param X_T: torch.tensor
        :param E_T: torch.tensor
        :param node_mask: torch.tensor
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        z_T = utils.PlaceHolder(X=X_T, E=E_T, y=y).type_as(X_T).mask(node_mask)

        # Start sampling from input X_T, E_T
        X, E, y = z_T.X, z_T.E, z_T.y  # (bs, n, dT), (bs, n, n, dT)

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T

        # NOTE : Always keep the first frame
        chain_X_size = torch.Size(
            (number_chain_steps, keep_chain, X.size(1))
        )  # e.g. (50, 30, n)
        chain_E_size = torch.Size(
            (number_chain_steps, keep_chain, E.size(1), E.size(2))
        )  # e.g. (50, 30, n, n)

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Make n_nodes tensor using X_T
        n_nodes = torch.sum(node_mask, dim=1)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        # NOTE : q(x_0 | x_1, x_0) = 1
        for s_int in reversed(range(1, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            # TODO : nll addition
            sampled_s, discrete_sampled_s = self.sample_bridge_p_zs_given_zt(
                s_norm, t_norm, X, E, X_T, E_T, y, node_mask
            )
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        gen_X, gen_E = sampled_s.X, sampled_s.E
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            # Overwrite last frame with the resulting X, E
            chain_X[0] = final_X_chain
            chain_E[0] = final_E_chain

            z_T_discrete = (
                utils.PlaceHolder(X=X_T, E=E_T, y=y)
                .type_as(X_T)
                .mask(node_mask, collapse=True)
            )

            chain_X[-1] = z_T_discrete.X[:keep_chain]
            chain_E[-1] = z_T_discrete.E[:keep_chain]

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []

        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        try:
            if self.visualization_tools is not None:
                self.print("Visualizing chains...")
                current_path = os.getcwd()
                num_molecules = chain_X.size(1)  # number of molecules
                for i in range(num_molecules):
                    result_path = os.path.join(
                        current_path,
                        f"chains/{self.cfg.general.name}/"
                        f"epoch{self.current_epoch}/"
                        f"chains/molecule_{batch_id + i}",
                    )
                    if not os.path.exists(result_path):
                        os.makedirs(result_path, exist_ok=True)
                        _ = self.visualization_tools.visualize_chain(
                            result_path,
                            chain_X[:, i, :].numpy(),
                            chain_E[:, i, :].numpy(),
                        )
                    self.print("\r{}/{} complete".format(i + 1, num_molecules))
                self.print("\nVisualizing molecules...")

                # Visualize the final molecules
                current_path = os.getcwd()
                result_path = os.path.join(
                    current_path,
                    f"graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/",
                )
                self.visualization_tools.visualize(
                    result_path, molecule_list, save_final
                )
                self.print("Done.")
        except Exception as e:
            print(f"Debug] rank: {self.local_rank} Error in visualization : {e}")

        return molecule_list, gen_X, gen_E

    def sample_bridge_p_zs_given_zt(self, s, t, X_t, E_t, X_T, E_T, y_t, node_mask):
        """Samples from p(x_s|x_t). Only used during sampling.
        Args:
            s (torch.Tensor): Time s (t-1)
            t (torch.Tensor): Time t
            X_t (torch.Tensor): Node features at time t
            E_t (torch.Tensor): Edge features at time t
            X_T (torch.Tensor): Node features at time T
            E_T (torch.Tensor): Edge features at time T
            y_t (torch.Tensor): Target labels at time t
            node_mask (torch.Tensor): Node mask
        Returns:
            out_one_hot (torch.Tensor): One-hot encoded sampled features
            out_discrete (torch.Tensor): Discrete sampled features
        """
        bs, n, dxs = X_t.shape

        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)

        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)  # (bs, 1)

        Qt = self.transition_model.get_Qt(beta_t, self.device)  # (bs, n ,d_t, d_t+1)

        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=self.device)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }
        extra_data = self.compute_extra_pair_data(noisy_data)
        # Masked PlaceHolder
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # (bs, n, d0)
        pred_E = F.softmax(pred.E, dim=-1)  # (bs, n, n, d0)

        pred_X = diffusion_utils.compute_backward_distribution(
            X_t, pred_X, Qt.X, Qsb.X, Qtb.X
        )
        pred_E = diffusion_utils.compute_backward_distribution(
            E_t, pred_E, Qt.E, Qsb.E, Qtb.E
        )
        pred_E = pred_E.reshape((bs, n, n, -1))

        sampled_s = diffusion_utils.sample_discrete_features(
            pred_X, pred_E, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return (
            out_one_hot.mask(node_mask).type_as(y_t),
            out_discrete.mask(node_mask, collapse=True).type_as(y_t),
        )

    def sample_bridge_p_zu_given_zt(self, t, u, X_t, E_t, y_t, node_mask):
        """Samples from p(x_u|x_t). Only used during sampling.
        Args:
            t (torch.Tensor): Time t
            u (torch.Tensor): Time u (t+1)
            X_t (torch.Tensor): Node features at time t
            E_t (torch.Tensor): Edge features at time t
            y_t (torch.Tensor): Target labels at time t
            node_mask (torch.Tensor): Node mask
        Returns:
            out_one_hot (torch.Tensor): One-hot encoded sampled features
            out_discrete (torch.Tensor): Discrete sampled features
        """
        bs, n, dxs = X_t.shape
        # t = t_array / self.T (0.0,0.1,...,0.9)
        # u = u_array / self.T (0.1,0.2,...,1.0)

        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }

        extra_data = self.compute_extra_pair_data(noisy_data)
        # Masked PlaceHolder
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # (bs, n, dT)
        pred_E = F.softmax(pred.E, dim=-1)  # (bs, n, n, dT)

        T_float = torch.ones(t.size()).float()

        beta_u = self.noise_schedule(t_normalized=u)  # (bs, 1)

        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)  # (bs, 1)
        alpha_u_bar = self.noise_schedule.get_alpha_bar(t_normalized=u)  # (bs, 1)
        alpha_T_bar = self.noise_schedule.get_alpha_bar(t_normalized=T_float)  # (bs, 1)

        Qu = self.transition_model.get_Qt(beta_u, self.device)  # (bs, n ,d_t, d_t+1)
        QuTb = self.transition_model.get_Qst_bar(
            alpha_t_bar, alpha_T_bar, device=self.device
        )
        QvTb = self.transition_model.get_Qst_bar(
            alpha_u_bar, alpha_T_bar, device=self.device
        )

        pred_X = diffusion_utils.compute_forward_distribution(
            X_t, pred_X, Qu.X, QuTb.X, QvTb.X
        )
        pred_E = diffusion_utils.compute_forward_distribution(
            E_t, pred_E, Qu.E, QuTb.E, QvTb.E
        )
        pred_E = pred_E.reshape((bs, n, n, -1))

        sampled_u = diffusion_utils.sample_discrete_features(
            probX=pred_X, probE=pred_E, node_mask=node_mask
        )

        X_u = F.one_hot(sampled_u.X, num_classes=self.Xdim_output).float()
        E_u = F.one_hot(sampled_u.E, num_classes=self.Edim_output).float()

        assert (E_u == torch.transpose(E_u, 1, 2)).all()
        assert (X_t.shape == X_u.shape) and (E_t.shape == E_u.shape)

        out_one_hot = utils.PlaceHolder(X=X_u, E=E_u, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_u, E=E_u, y=torch.zeros(y_t.shape[0], 0))

        return (
            out_one_hot.mask(node_mask).type_as(y_t),
            out_discrete.mask(node_mask, collapse=True).type_as(y_t),
        )

    def compute_extra_pair_data(self, noisy_data):
        # TODO : Implement extra_features and domain_features !!!
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(
            noisy_data
        )  # charge, valency, weight

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
