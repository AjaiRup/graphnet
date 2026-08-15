"""Train IceMix for joint neutrino direction and vertex reconstruction.

Two prediction heads share one `DeepIce` backbone: a vertex head trained with a
Euclidean distance loss and a direction head trained with a 3D von Mises-Fisher
loss, combined as `alpha * vertex + direction`. An optional second stage
(`--stage2-from`) fits a per-event vertex uncertainty on the frozen network.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graphnet.constants import DATASETS_CONFIG_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.dataloader import DataLoader, do_shuffle
from graphnet.data.dataset import EnsembleDataset, LengthGroupedSampler
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.models import StandardModel
from graphnet.models.data_representation import GraphDefinition
from graphnet.models.data_representation.graphs.nodes import IceMixNodes
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DeepIce
from graphnet.models.task.task import IdentityTask
from graphnet.models.task.reconstruction import (
    DirectionReconstructionWithKappaDecoupled,
    PositionReconstructionWithSigma,
)
from graphnet.training.callbacks import (
    PhysicsMetricsCallback,
    ProgressBar,
    RandomRotationCallback,
)
from graphnet.training.labels import Direction
from graphnet.training.loss_functions import (
    EnsembleLoss,
    EuclideanDistanceLoss,
    IsotropicGaussianNLLLoss,
    PerEventRecordingLoss,
    VonMisesFisher3DLoss,
)
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import DatasetConfig, ModelConfig
from graphnet.utilities.logging import Logger
from graphnet.data.utilities.lengths import event_lengths
from graphnet.data.utilities.paths import resolve_dataset_paths
from graphnet.data.utilities.splits import seeded_event_splits

POSITION_LABELS = ["position_x", "position_y", "position_z"]
FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "hlc"]

CURSOR_PREDICTION_ALIASES = {
    "position_x_pred": "pos_x_pred",
    "position_y_pred": "pos_y_pred",
    "position_z_pred": "pos_z_pred",
    "direction_kappa": "dir_kappa_pred",
}

# Metres per unit of the standardized position targets.
POSITION_SCALE_M = 100.0

# Fixed hyperparameters of the validated configuration.
MLP_RATIO = 4
DROPOUT = 0.1
ATTN_DROP = 0.05
PROJ_DROP = 0.1
DROP_PATH_RATE = 0.2
ALPHA = 2.6
KAPPA_SWITCH = 700.0
SEL_ENERGY_MIN_GEV = 10.0
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5


def model_size(smoke: bool) -> Dict[str, int]:
    """Return the transformer dimensions, shrunk for the CPU smoke test."""
    if smoke:
        return dict(
            hidden_dim=32, seq_length=16, depth=2, depth_rel=1, n_rel=1,
            head_size=8,
        )
    return dict(
        hidden_dim=384, seq_length=128, depth=12, depth_rel=4, n_rel=4,
        head_size=32,
    )


def scale_position(x):  # type: ignore[no-untyped-def]
    # Keep this self-contained: ModelConfig serializes the function body.
    x[:, 0] = (x[:, 0] - 46.3) / 100.0
    x[:, 1] = (x[:, 1] + 34.9) / 100.0
    x[:, 2] = (x[:, 2] + 350.0) / 100.0
    return x


def unscale_position(x):  # type: ignore[no-untyped-def]
    x[:, 0] = x[:, 0] * 100.0 + 46.3
    x[:, 1] = x[:, 1] * 100.0 - 34.9
    x[:, 2] = x[:, 2] * 100.0 - 350.0
    return x


def unscale_position_sigma(x):  # type: ignore[no-untyped-def]
    x[:, 0] = x[:, 0] * 100.0 + 46.3
    x[:, 1] = x[:, 1] * 100.0 - 34.9
    x[:, 2] = x[:, 2] * 100.0 - 350.0
    x[:, 3] = x[:, 3].exp() * 100.0
    return x


def add_cursor_prediction_aliases(results):  # type: ignore[no-untyped-def]
    """Add the prediction names consumed by the cursor plotting scripts."""
    for source, target in CURSOR_PREDICTION_ALIASES.items():
        results[target] = results[source]
    return results


class FrozenTrunkEvalCallback(Callback):
    """Keep the frozen Stage-1 network in evaluation mode."""

    def _pin(self, pl_module: Any) -> None:
        pl_module.backbone.eval()

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._pin(pl_module)

    def on_train_batch_start(
        self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int
    ) -> None:
        self._pin(pl_module)


def single_threaded_worker(_worker_id: int) -> None:
    """Pin a dataloader worker to one thread."""
    torch.set_num_threads(1)


def batch_limit(value: str) -> Any:
    """Parse a `--limit-*-batches` value: `int` is a batch count, `float`
    a fraction of the epoch."""
    return float(value) if "." in value else int(value)


def build_data_representation(
    features: List[str],
    max_pulses: int,
    hlc_name: Optional[str],
    add_ice_properties: bool,
    add_static_features: bool = False,
) -> GraphDefinition:
    """Return the pulse-sequence representation IceMix consumes."""
    return GraphDefinition(
        detector=IceCube86(),
        node_definition=IceMixNodes(
            input_feature_names=features,
            max_pulses=max_pulses,
            z_name="dom_z",
            hlc_name=hlc_name,
            add_ice_properties=add_ice_properties,
        ),
        input_feature_names=features,
        add_static_features=add_static_features,
    )


def ddp_ranks(n_devices: int) -> Tuple[int, int]:
    """Return `(rank, world_size)` before DDP has started.

    `LengthGroupedSampler` shards the epoch at construction time, so the
    rank must come from SLURM variables, not the not-yet-created process
    group.
    """
    tasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if n_devices <= 1:
        return 0, 1
    if tasks != n_devices:
        raise RuntimeError(
            f"Training on {n_devices} GPUs needs {n_devices} SLURM tasks, but "
            f"SLURM_NTASKS={tasks}. Launch the run under `srun "
            f"--ntasks={n_devices} --ntasks-per-node={n_devices}`, with the "
            "sbatch header requesting the same number of tasks. For an "
            "interactive allocation, request them there too: "
            f"`salloc -N1 --ntasks-per-node={n_devices} --gres=gpu:{n_devices}"
            "`. A single-GPU run (`--gpus 0`) needs none of this."
        )
    return int(os.environ["SLURM_PROCID"]), tasks


def build_dataloaders(
    dataset_config: DatasetConfig,
    data_representation: GraphDefinition,
    dataloader_kwargs: Dict[str, Any],
    split_ratio: List[float],
    split_cache_dir: Optional[str],
    group_factor: int = 0,
    length_cache_dir: Optional[str] = None,
    max_pulses: Optional[int] = None,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, DataLoader]:
    """Return the train/validation/test dataloaders.

    A dataset config with a `selection` is delegated to
    `DataLoader.from_dataset_config`; otherwise the split comes from
    `seeded_event_splits` and batches are length-matched by
    `LengthGroupedSampler`.
    """
    logger = Logger()
    if dataset_config.selection is not None:
        if group_factor > 0:
            logger.warning(
                "Length grouping needs the per-database index lists that "
                "`seeded_event_splits` produces, and the dataset config "
                "carries a `selection`. Falling back to random batching."
            )
        logger.info("Resolving splits from the dataset config's `selection`.")
        dataloaders = DataLoader.from_dataset_config(
            dataset_config, **dataloader_kwargs
        )
        assert isinstance(dataloaders, dict)
        return dataloaders

    paths = resolve_dataset_paths(dataset_config.path)
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Dataset path(s) do not exist after resolving $GRAPHNET and "
            f"environment variables: {missing}. Is DATA_ROOT set?"
        )

    logger.info(
        f"No `selection` in the dataset config; splitting {len(paths)} "
        f"database(s) with seed={dataset_config.seed} ratio={split_ratio}."
    )
    splits = seeded_event_splits(
        paths=paths,
        seed=dataset_config.seed if dataset_config.seed is not None else 42,
        ratio=split_ratio,
        index_column=dataset_config.index_column,
        truth_table=dataset_config.truth_table,
        cache_dir=split_cache_dir,
    )

    shared: Dict[str, Any] = dict(
        pulsemaps=dataset_config.pulsemaps,
        features=dataset_config.features,
        truth=dataset_config.truth,
        truth_table=dataset_config.truth_table,
        index_column=dataset_config.index_column,
        data_representation=data_representation,
        labels={
            "direction": Direction(azimuth_key="azimuth", zenith_key="zenith")
        },
        add_default_labels=False,
    )

    dataloaders = {}
    for name, per_database in splits.items():
        # The pulse-count array must align with the datasets actually used.
        datasets, used_paths, used_selections = [], [], []
        for path, selection in zip(paths, per_database):
            if not selection:
                continue
            datasets.append(
                SQLiteDataset(path=path, selection=selection, **shared)
            )
            used_paths.append(path)
            used_selections.append(selection)
        if not datasets:
            continue

        # `test` keeps a sequential sampler: `predict_as_dataframe` re-walks
        # the loader to align per-event attributes.
        if group_factor > 0 and name != "test":
            lengths = event_lengths(
                paths=used_paths,
                selections=used_selections,
                pulsemap=dataset_config.pulsemaps[0],
                index_column=dataset_config.index_column,
                cache_dir=length_cache_dir,
                max_pulses=max_pulses,
            )
            logger.info(
                f"Length-grouping the '{name}' split at "
                f"group_factor={group_factor} over {world_size} rank(s)."
            )
            dataloaders[name] = DataLoader(
                EnsembleDataset(datasets),
                sampler=LengthGroupedSampler(
                    lengths=lengths,
                    batch_size=dataloader_kwargs["batch_size"],
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=do_shuffle(name),
                    seed=seed,
                    group_factor=group_factor,
                ),
                # Drop the training remainder; validation keeps its full
                # split.
                drop_last=do_shuffle(name),
                **dataloader_kwargs,
            )
            continue

        dataloaders[name] = DataLoader(
            EnsembleDataset(datasets),
            shuffle=do_shuffle(name),
            **dataloader_kwargs,
        )
    return dataloaders


def build_model(args: Any) -> StandardModel:
    """Return the `StandardModel` to train."""
    smoke = getattr(args, "smoke_test", False)
    data_representation = build_data_representation(
        features=FEATURES,
        max_pulses=32 if smoke else 256,
        hlc_name="hlc",
        add_ice_properties=False,
    )

    if args.model_config:
        Logger().info(f"Loading model from {args.model_config}")
        model = StandardModel.from_config(
            ModelConfig.load(args.model_config), trust=True
        )
        assert isinstance(model, StandardModel)
        return model

    backbone = DeepIce(
        mlp_ratio=MLP_RATIO,
        scaled_emb=True,
        include_dynedge=False,
        n_features=len(FEATURES),
        init_values=1.0,
        init_values_rel=1.0,
        dropout=DROPOUT,
        attn_drop=ATTN_DROP,
        proj_drop=PROJ_DROP,
        drop_path_rate=DROP_PATH_RATE,
        norm_cls=True,
        **model_size(smoke),
    )

    if args.stage2_from:
        position_task = PositionReconstructionWithSigma(
            hidden_size=backbone.nb_outputs,
            target_labels=POSITION_LABELS,
            loss_function=PerEventRecordingLoss(
                IsotropicGaussianNLLLoss(detach_position=True),
                metric_name="position",
            ),
            transform_target=scale_position,
            transform_inference=unscale_position_sigma,
        )
    else:
        position_task = IdentityTask(
            nb_outputs=3,
            target_labels=POSITION_LABELS,
            prediction_labels=[
                "position_x_pred",
                "position_y_pred",
                "position_z_pred",
            ],
            hidden_size=backbone.nb_outputs,
            loss_function=PerEventRecordingLoss(
                EnsembleLoss(
                    loss_functions=[EuclideanDistanceLoss()],
                    loss_factors=[ALPHA],
                ),
                metric_name="position",
            ),
            transform_target=scale_position,
            transform_inference=unscale_position,
        )

    tasks = [
        position_task,
        DirectionReconstructionWithKappaDecoupled(
            hidden_size=backbone.nb_outputs,
            target_labels=["direction"],
            loss_function=PerEventRecordingLoss(
                VonMisesFisher3DLoss(
                    kappa_switch=KAPPA_SWITCH,
                    approximation="tight",
                ),
                metric_name="direction",
            ),
        ),
    ]

    optimizer_kwargs: Dict[str, Any] = {"lr": args.lr, "eps": 1e-05}
    if args.stage2_from:
        # AdamW's decoupled weight decay updates every tensor with a
        # gradient, including the grafted position rows; zero decay keeps
        # Stage-1 outputs bit-exact.
        optimizer_kwargs["weight_decay"] = 0.0

    return StandardModel(
        data_representation=data_representation,
        backbone=backbone,
        tasks=tasks,
        optimizer_class=AdamW,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={
            "mode": "min",
            "patience": SCHEDULER_PATIENCE,
            "factor": SCHEDULER_FACTOR,
        },
        scheduler_config={"monitor": "val_loss", "frequency": 1},
    )


def build_callbacks(args: Any, output_dir: str, rank: int = 0) -> List[Any]:
    """Return the training callbacks."""
    callbacks: List[Any] = [
        ProgressBar(),
        PhysicsMetricsCallback(
            position_scale=POSITION_SCALE_M,
            sel_energy_min=SEL_ENERGY_MIN_GEV,
        ),
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        # `val_loss` is direction-dominated; track the vertex-best epoch
        # separately.
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="best-position-{epoch:02d}",
            auto_insert_metric_name=False,
            monitor="val/position_error_m_mean",
            mode="min",
            save_top_k=1,
        ),
    ]
    # Track the direction-best epoch on the in-scope energy subset.
    callbacks.append(
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="best-direction-{epoch:02d}",
            auto_insert_metric_name=False,
            monitor="val/angular_error_deg_sel_median",
            mode="min",
            save_top_k=1,
        )
    )
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=args.early_stopping_patience,
                mode="min",
            )
        )
    if args.stage2_from:
        callbacks.append(FrozenTrunkEvalCallback())
    if args.rotation_augmentation:
        # Offset by rank so the ranks don't all draw the same angle stream
        # while still being reproducible from `--seed`.
        callbacks.append(
            RandomRotationCallback(
                seed=None if args.seed is None else args.seed + rank
            )
        )
    return callbacks


def main(args: Any) -> None:
    """Train the model and write predictions to `--output-dir`."""
    logger = Logger()
    if args.seed is not None:
        from pytorch_lightning import seed_everything

        seed_everything(args.seed, workers=True)

    if args.save_model_config:
        model = build_model(args)
        model.save_config(args.save_model_config)
        logger.info(f"Wrote model config to {args.save_model_config}")
        return

    dataset_config = DatasetConfig.load(args.dataset_config)

    output_dir = args.output_dir or os.path.join(
        EXAMPLE_OUTPUT_DIR, "icemix_joint_reconstruction", args.run_name
    )
    if rank_zero_only.rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    max_pulses = 32 if args.smoke_test else 256
    data_representation = build_data_representation(
        features=FEATURES,
        max_pulses=max_pulses,
        hlc_name="hlc",
        add_ice_properties=False,
    )

    dataloader_kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.gpus),
    }
    if args.num_workers > 0:
        dataloader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=4,
            worker_init_fn=single_threaded_worker,
        )
    else:
        dataloader_kwargs.update(
            persistent_workers=False, prefetch_factor=None
        )

    rank, world_size = ddp_ranks(len(args.gpus) if args.gpus else 1)

    dataloaders = build_dataloaders(
        dataset_config=dataset_config,
        data_representation=data_representation,
        dataloader_kwargs=dataloader_kwargs,
        split_ratio=[0.6, 0.2, 0.2] if args.smoke_test else [0.8, 0.1, 0.1],
        split_cache_dir=args.split_cache_dir or args.cache_dir,
        group_factor=args.group_factor,
        length_cache_dir=args.length_cache_dir or args.cache_dir,
        max_pulses=max_pulses,
        seed=args.seed if args.seed is not None else 42,
        rank=rank,
        world_size=world_size,
    )
    logger.info(
        "Dataloader sizes: "
        + ", ".join(
            f"{name}={len(loader.dataset)}"
            for name, loader in dataloaders.items()
        )
    )

    model = build_model(args)

    if args.fine_tune_from and args.stage2_from:
        raise ValueError(
            "--fine-tune-from and --stage2-from are mutually exclusive: "
            "fine-tuning trains everything, Stage 2 trains only the sigma "
            "row of the position head."
        )

    if args.fine_tune_from:
        logger.info(f"Loading weights for fine-tuning: {args.fine_tune_from}")
        checkpoint = torch.load(
            args.fine_tune_from, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))

    if args.stage2_from:
        logger.info(f"Stage 2: grafting Stage-1 weights: {args.stage2_from}")
        checkpoint = torch.load(
            args.stage2_from, map_location="cpu", weights_only=False
        )
        state = checkpoint.get("state_dict", checkpoint)
        weight = state["_tasks.0._affine.weight"]
        bias = state["_tasks.0._affine.bias"]
        if weight.shape[0] == 3:
            # Graft the 3-row Stage-1 head; sigma starts at log(sigma) = 0.
            expanded_weight = torch.zeros(4, weight.shape[1])
            expanded_weight[:3] = weight
            expanded_bias = torch.zeros(4)
            expanded_bias[:3] = bias
            state["_tasks.0._affine.weight"] = expanded_weight
            state["_tasks.0._affine.bias"] = expanded_bias
        model.load_state_dict(state)

        # Freeze everything except the position head; the grafted rows
        # stay bit-identical (detached loss, zero weight decay).
        trainable = 0
        for name, parameter in model.named_parameters():
            is_position_head = name.startswith("_tasks.0._affine")
            parameter.requires_grad_(is_position_head)
            trainable += parameter.numel() if is_position_head else 0
        logger.info(
            f"Stage 2: {trainable} trainable parameters (position head)."
        )

    loggers: List[Any] = [
        CSVLogger(save_dir=os.path.join(output_dir, "logs"), name="training")
    ]
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            save_dir=os.path.join(output_dir, "logs"),
            name=args.run_name,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
        )
        # Rank zero only: concurrent writes race.
        if rank_zero_only.rank == 0:
            wandb_logger.experiment.config.update(
                {**vars(args), "dataset_config_path": args.dataset_config},
                allow_val_change=True,
            )
        loggers.append(wandb_logger)

    # Pin `SLURMEnvironment`: detection fails under `salloc`
    # (SLURM_JOB_NAME="bash"), and Lightning would then fork its own
    # children inside each srun task.
    extra_trainer_kwargs: Dict[str, Any] = {}
    if world_size > 1:
        extra_trainer_kwargs["plugins"] = [
            SLURMEnvironment(auto_requeue=False)
        ]
        logger.info(
            f"DDP over {len(args.gpus)} GPUs as {world_size} srun task(s), "
            f"this one rank {rank} (effective batch "
            f"{args.batch_size * world_size}; --lr is NOT auto-scaled)."
        )

    model.fit(
        dataloaders["train"],
        dataloaders.get("validation"),
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        gpus=args.gpus,
        callbacks=build_callbacks(args, output_dir, rank=rank),
        ckpt_path=args.resume_from,
        logger=loggers,
        distribution_strategy=(
            "ddp" if args.gpus and len(args.gpus) > 1 else "auto"
        ),
        precision="32-true" if args.smoke_test else "16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0,
        gradient_clip_val=args.gradient_clip_val,
        **extra_trainer_kwargs,
    )

    # Under DDP every rank runs the whole script, so an unguarded write here
    # is eight processes opening the same path at once. The content is
    # identical, which makes a torn file the only observable symptom -- and a
    # rare one, which is worse. Rank zero writes; the rest already hold the
    # same weights and have nothing to contribute.
    if rank_zero_only.rank == 0:
        model.save_state_dict(os.path.join(output_dir, "state_dict.pth"))
        model.save_config(os.path.join(output_dir, "model_config.yml"))

    if args.skip_prediction:
        logger.info("--skip-prediction given; not running inference.")
        return

    prediction_loader = dataloaders.get("test") or dataloaders.get(
        "validation"
    )
    if prediction_loader is None:
        logger.warning("No held-out split; skipping prediction.")
        return

    wanted_attributes = [
        *POSITION_LABELS,
        "azimuth",
        "zenith",
        "energy",
        "pid",
        "interaction_type",
        "oneweight",
        dataset_config.index_column,
    ]
    results = model.predict_as_dataframe(
        prediction_loader,
        additional_attributes=wanted_attributes,
        gpus=args.gpus,
        # Sharding the test loader would misalign predictions with the
        # attribute pass inside `predict_as_dataframe`.
        use_distributed_sampler=False,
    )

    missing = [
        attribute
        for attribute in wanted_attributes
        if attribute not in results.columns
    ]
    if missing:
        raise RuntimeError(
            f"Prediction dropped the requested attribute(s) {missing}, so the "
            f"{len(results)} row(s) cannot be joined back to truth. This "
            "means the prediction dataloader was resampled or sharded; check "
            "that the `test` split still uses a plain `SequentialSampler`."
        )

    if rank_zero_only.rank == 0:
        predictions_dir = os.path.join(output_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        results = add_cursor_prediction_aliases(results)
        results.to_csv(
            os.path.join(predictions_dir, "results.csv"), index=False
        )
        logger.info(f"Wrote predictions to {predictions_dir}/results.csv")


def _apply_smoke_test_overrides(args: Any) -> None:
    """Shrink the run so it completes on CPU against the test fixture."""
    args.dataset_config = os.path.join(
        DATASETS_CONFIG_DIR, "icecube86_joint_reconstruction_fixture.yml"
    )
    args.max_epochs = 2
    args.early_stopping_patience = 0
    args.batch_size = 2
    args.num_workers = 0
    args.gpus = None
    args.wandb = False
    args.output_dir = args.output_dir or os.path.join(
        EXAMPLE_OUTPUT_DIR, "icemix_joint_reconstruction", "smoke_test"
    )


def build_parser() -> ArgumentParser:
    """Return the argument parser, callable from external harnesses via
    `build_parser().parse_args([])`."""
    parser = ArgumentParser(
        description="Train IceMix for joint direction + vertex reconstruction."
    )
    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 100),
        ("early-stopping-patience", 12),
        ("batch-size", 256),
        ("num-workers", 3),
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset-config",
        default="configs/datasets/icecube86_joint_reconstruction.yml",
        help="Path to a `DatasetConfig` YAML (default: %(default)s)",
    )
    data.add_argument(
        "--group-factor",
        type=int,
        default=16,
        help="Length-grouping window in global steps; 0 uses random batches.",
    )
    data.add_argument(
        "--cache-dir",
        default=None,
        help="Default cache directory for splits and pulse counts.",
    )
    data.add_argument("--split-cache-dir", default=None)
    data.add_argument(
        "--length-cache-dir",
        default=None,
        help="Cache directory for per-event pulse counts.",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model-config", default=None)
    model_group.add_argument(
        "--save-model-config",
        default=None,
        help="Write the model config and exit.",
    )

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--lr", type=float, default=6.25e-5)
    optimisation.add_argument(
        "--gradient-clip-val",
        type=float,
        default=None,
        help="Maximum gradient norm.",
    )
    optimisation.add_argument("--accumulate-grad-batches", type=int, default=1)
    optimisation.add_argument(
        "--limit-train-batches",
        type=batch_limit,
        default=None,
        help="Batch count or decimal epoch fraction.",
    )
    optimisation.add_argument(
        "--limit-val-batches", type=batch_limit, default=None
    )
    optimisation.add_argument("--seed", type=int, default=42)
    optimisation.add_argument(
        "--rotation-augmentation",
        action="store_true",
        help="Apply z-axis rotation to training batches.",
    )

    run = parser.add_argument_group("run")
    run.add_argument("--run-name", default="icemix")
    run.add_argument("--output-dir", default=None)
    run.add_argument("--resume-from", default=None)
    run.add_argument("--fine-tune-from", default=None)
    run.add_argument(
        "--stage2-from",
        default=None,
        help="Calibrate vertex sigma on this frozen Stage-1 checkpoint.",
    )
    run.add_argument("--wandb", action="store_true")
    run.add_argument("--wandb-project", default="IceMix")
    run.add_argument("--wandb-entity", default=None)
    run.add_argument("--wandb-run-id", default=None)
    run.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip held-out prediction after training.",
    )
    run.add_argument(
        "--smoke-test",
        action="store_true",
        help="Shrink everything to run on CPU against the test fixture.",
    )

    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.smoke_test:
        _apply_smoke_test_overrides(parsed)
    main(parsed)
