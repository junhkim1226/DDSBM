import argparse
import os
import pathlib
import warnings

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import ddsbm
from ddsbm import utils
from ddsbm.analysis.visualization import MolecularVisualization
from ddsbm.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from ddsbm.diffusion.extra_features_molecular import ExtraMolecularFeatures
from ddsbm.diffusion_model_discrete import DiscreteDenoisingDiffusion
from ddsbm.metrics.abstract_metrics import (
    TrainAbstractMetrics,
    TrainAbstractMetricsDiscrete,
)
from ddsbm.metrics.molecular_metrics import SamplingMolecularMetrics
from ddsbm.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """Resumes a run. It loads previous config without allowing to update keys (used for testing)."""
    saved_cfg = cfg.copy()
    name = cfg.general.name + "_resume"
    resume = cfg.general.test_only
    gen_data_path = cfg.graph_match.data_path
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    # NOTE: saved_cfg: new cfg, cfg: previous cfg
    # This function overrides previous cfg's general part with new cfg
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.general.gpus = saved_cfg.general.gpus

    cfg.general.sample_batch_size = saved_cfg.general.sample_batch_size
    cfg.general.wandb = saved_cfg.general.wandb
    cfg.general.final_model_samples_to_generate = (
        saved_cfg.general.final_model_samples_to_generate
    )
    cfg.general.final_model_samples_to_save = (
        saved_cfg.general.final_model_samples_to_save
    )
    cfg.general.final_model_chains_to_save = (
        saved_cfg.general.final_model_chains_to_save
    )
    # cfg.model.min_alpha = saved_cfg.model.min_alpha  # NOTE: reset min_alpha
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    # NOTE: use pre-defiend data path
    cfg.graph_match.data_path = gen_data_path
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split("outputs")[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == "discrete":
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(
            resume_path, **model_kwargs
        )
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(
            resume_path, **model_kwargs
        )
    new_cfg = model.cfg

    # NOTE: cfg is new config (overrides previous config)
    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def train(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    dataset_name = dataset_config["name"]
    is_unconditional = dataset_config["unconditional"]

    if is_unconditional:
        if dataset_name == "uncond_qm9":
            from ddsbm.datasets.uncond_jointmol_dataset import (
                JointMolDataModule,
                JointMolecularinfos,
                get_train_smiles,
            )

            datamodule = JointMolDataModule(cfg)
            dataset_infos = JointMolecularinfos(datamodule, cfg)
            train_smiles = get_train_smiles(
                cfg=cfg,
                train_dataloader=datamodule.train_dataloader(),
                dataset_infos=dataset_infos,
                evaluate_dataset=False,
            )

            if cfg.model.extra_features is not None:
                extra_features = ExtraFeatures(
                    cfg.model.extra_features, dataset_info=dataset_infos
                )
                domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
            else:
                extra_features = DummyExtraFeatures()
                domain_features = DummyExtraFeatures()

            dataset_infos.compute_input_output_dims(
                datamodule=datamodule,
                extra_features=extra_features,
                domain_features=domain_features,
            )

            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

            # We do not evaluate novelty during training
            sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
            visualization_tools = MolecularVisualization(
                cfg.dataset.remove_h, dataset_infos=dataset_infos
            )
        elif dataset_name == "uncond_comm20":
            from ddsbm.analysis.spectre_utils import Comm20SamplingMetrics
            from ddsbm.analysis.visualization import NonMolecularVisualization
            from ddsbm.datasets.uncond_comm20_dataset import (
                Comm20DataModule,
                Comm20infos,
            )

            datamodule = Comm20DataModule(cfg)
            dataset_infos = Comm20infos(datamodule, cfg)

            train_metrics = TrainAbstractMetricsDiscrete()
            sampling_metrics = Comm20SamplingMetrics(
                datamodule, direction=cfg.train.bridge_direction
            )
            visualization_tools = NonMolecularVisualization()

            extra_features = ExtraFeatures(
                cfg.model.extra_features, dataset_info=dataset_infos
            )
            domain_features = DummyExtraFeatures()

            dataset_infos.compute_input_output_dims(
                datamodule=datamodule,
                extra_features=extra_features,
                domain_features=domain_features,
            )
        elif dataset_name == "uncond_planar":
            from ddsbm.analysis.spectre_utils import PlanarSamplingMetrics
            from ddsbm.analysis.visualization import NonMolecularVisualization
            from ddsbm.datasets.uncond_planar_dataset import (
                PlanarDataModule,
                Planarinfos,
            )

            datamodule = PlanarDataModule(cfg)
            dataset_infos = Planarinfos(datamodule, cfg)

            train_metrics = TrainAbstractMetricsDiscrete()
            sampling_metrics = PlanarSamplingMetrics(
                datamodule, direction=cfg.train.bridge_direction
            )
            visualization_tools = NonMolecularVisualization()

            extra_features = ExtraFeatures(
                cfg.model.extra_features, dataset_info=dataset_infos
            )
            domain_features = DummyExtraFeatures()

            dataset_infos.compute_input_output_dims(
                datamodule=datamodule,
                extra_features=extra_features,
                domain_features=domain_features,
            )
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported for unconditional training."
            )
    else:
        from ddsbm.datasets.jointmol_dataset import (
            JointMolDataModule,
            JointMolecularinfos,
            get_train_smiles,
        )

        datamodule = JointMolDataModule(cfg)
        dataset_infos = JointMolecularinfos(datamodule, cfg)
        train_smiles = get_train_smiles(
            cfg=cfg,
            train_dataloader=datamodule.train_dataloader(),
            dataset_infos=dataset_infos,
            evaluate_dataset=False,
            direction=cfg.train.bridge_direction,
        )

        if cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(
                cfg.model.extra_features, dataset_info=dataset_infos
            )
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(
            cfg.dataset.remove_h, dataset_infos=dataset_infos
        )

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
    }

    seed = cfg.general.seed
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if cfg.train.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.general.test_only:
        # NOTE: When testing, previous configuration is fully loaded
        # Also, we change working directory before executing train_helper.py
        cfg, _ = get_resume(cfg, model_kwargs)
        cfg.general.seed = seed  # NOTE: switch to pre-defined seed
        os.makedirs(".hydra", exist_ok=True)
        OmegaConf.save(cfg, f".hydra/test_config_seed_{seed}.yaml", resolve=True)
    elif cfg.general.resume is not None:
        # NOTE: When resuming for train, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)

    utils.create_folders(cfg)

    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            # dirpath=f"checkpoints/{cfg.general.name}",
            dirpath="checkpoints",
            filename="{epoch}",
            monitor="val/epoch_NLL",
            save_top_k=-1,
            every_n_epochs=cfg.general.save_every_n_epochs,
            mode="min",
        )
        last_ckpt_save = ModelCheckpoint(
            # dirpath=f"checkpoints/{cfg.general.name}",
            dirpath="checkpoints",
            filename="last",
            every_n_epochs=1000000000,  # NOTE: Save last model only at the end of training
            save_last=True,
            save_on_train_epoch_end=True,
        )
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    name = cfg.general.name
    if name == "debug":
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == "debug",
        enable_progress_bar=True,
        callbacks=callbacks,
        log_every_n_steps=50 if name != "debug" else 1,
        logger=[],
    )

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        trainer.predict(model, datamodule=datamodule)
    else:
        if is_unconditional and dataset_name in ["uncond_comm20", "uncond_planar"]:
            trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        else:
            trainer.predict(
                model,
                dataloaders=datamodule.test_dataloader(),
                ckpt_path=cfg.general.test_only,
            )
    trainer.strategy.teardown()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train(cfg)
