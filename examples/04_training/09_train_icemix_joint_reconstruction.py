"""Train IceMix for joint direction and vertex reconstruction."""

import os
import json
import argparse
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from graphnet.constants import (
    DATASETS_CONFIG_DIR,
    EXAMPLE_OUTPUT_DIR,
    GRAPHNET_ROOT_DIR,
)
from graphnet.data.dataloader import DataLoader, do_shuffle
from graphnet.data.dataset import EnsembleDataset, LengthGroupedSampler
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.models import StandardModel
from graphnet.models.data_representation import GraphDefinition
from graphnet.training.callbacks import (
    PhysicsMetricsCallback,
    ProgressBar,
    RandomRotationCallback,
)
from graphnet.training.labels import Direction
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import DatasetConfig
from graphnet.utilities.logging import Logger
from graphnet.data.utilities.lengths import event_lengths
from graphnet.data.utilities.paths import resolve_dataset_paths
from graphnet.data.utilities.splits import seeded_event_splits

from graphnet.models.transformer.icemix import (
    POSITION_LABELS,
    build_data_representation,
    load_icemix_model,
    validate_dataset_config,
)

POSITION_SCALE_M = 100.0
# Rotate nodes and global targets about the same physical DeepCore axis.
DEEP_CORE_XY_ORIGIN_M = (46.3, -34.9)
ICECUBE86_XY_SCALE_M = 500.0
SEL_ENERGY_MIN_GEV = 10.0
MODEL_TUNING_ARGUMENTS = (
    "vertex_weight",
    "direction_weight",
    "lr",
    "weight_decay",
    "adam_eps",
    "scheduler_factor",
    "scheduler_patience",
    "dropout",
    "attn_drop",
    "proj_drop",
    "drop_path_rate",
)


def single_threaded_worker(_worker_id: int) -> None:
    """Pin a dataloader worker to one thread."""
    torch.set_num_threads(1)


def batch_limit(value: str) -> Any:
    """Parse a `--limit-*-batches` value: `int` is a batch count, `float`
    a fraction of the epoch."""
    return float(value) if "." in value else int(value)


def ddp_ranks(n_devices: int) -> Tuple[int, int]:
    """Return `(rank, world_size)` before DDP has started."""
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
    """Return the train/validation/test dataloaders."""
    logger = Logger()
    validate_dataset_config(dataset_config)
    if dataset_config.selection is not None:
        representation = data_representation.config.dict()
        representation["arguments"]["dtype"] = "torch.float32"
        dataset_config.data_representation = representation
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
    return load_icemix_model(
        args.model_config,
        **{name: getattr(args, name, None) for name in MODEL_TUNING_ARGUMENTS},
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
        # Keep task-specific checkpoints alongside the joint-loss checkpoint.
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
    if args.rotation_augmentation:
        position_xy_origin = DEEP_CORE_XY_ORIGIN_M
        node_xy_origin = tuple(
            coordinate / ICECUBE86_XY_SCALE_M
            for coordinate in position_xy_origin
        )
        callbacks.append(
            RandomRotationCallback(
                seed=None if args.seed is None else args.seed + rank,
                node_xy_origin=node_xy_origin,
                position_xy_origin=position_xy_origin,
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
    validate_dataset_config(dataset_config)

    output_dir = args.output_dir or os.path.join(
        EXAMPLE_OUTPUT_DIR, "icemix_joint_reconstruction", args.run_name
    )
    if rank_zero_only.rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    max_pulses = 256
    data_representation = build_data_representation()

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

    if rank_zero_only.rank == 0:
        # Keep the exact architecture inspectable even if this run is canceled.
        model.save_config(os.path.join(output_dir, "model_config.yml"))

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
            DDPStrategy(timeout=timedelta(minutes=10))
            if args.gpus and len(args.gpus) > 1
            else "auto"
        ),
        precision="32-true" if args.smoke_test else "16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0,
        gradient_clip_val=args.gradient_clip_val,
        **extra_trainer_kwargs,
    )

    # Only rank zero writes shared artifacts.
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
        *(["oneweight"] if "oneweight" in dataset_config.truth else []),
        dataset_config.index_column,
        "n_pulses",
    ]
    results = model.predict_as_dataframe(
        prediction_loader,
        additional_attributes=wanted_attributes,
        gpus=args.gpus,
        # Sharding the test loader would misalign predictions with the
        # attribute pass inside `predict_as_dataframe`.
        use_distributed_sampler=False,
        precision="32-true",
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

    datasets = getattr(
        prediction_loader.dataset, "datasets", [prediction_loader.dataset]
    )
    sources = [
        os.path.realpath(dataset.path)
        for dataset in datasets
        for _ in range(len(dataset))
    ]
    if len(sources) != len(results):
        raise RuntimeError(
            "Prediction row count differs from the selected events"
        )
    results["source_db"] = sources

    if rank_zero_only.rank == 0:
        predictions_dir = os.path.join(output_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        results.to_csv(
            os.path.join(predictions_dir, "results.csv"), index=False
        )
        with open(
            os.path.join(predictions_dir, "metadata.json"), "w"
        ) as handle:
            json.dump(
                {
                    "checkpoint": os.path.abspath(
                        os.path.join(output_dir, "state_dict.pth")
                    ),
                    "selection": "last epoch",
                    "precision": "32-true",
                    "seed": args.seed,
                    "split_seed": dataset_config.seed,
                    "dataset_config": os.path.abspath(args.dataset_config),
                    "split": "test" if "test" in dataloaders else "validation",
                    "model_config": os.path.abspath(
                        os.path.join(output_dir, "model_config.yml")
                    ),
                },
                handle,
                indent=2,
            )
        logger.info(f"Wrote predictions to {predictions_dir}/results.csv")


def _apply_smoke_test_overrides(args: Any) -> None:
    """Limit data and steps while retaining the full model."""
    args.dataset_config = os.path.join(
        DATASETS_CONFIG_DIR, "icecube86_joint_reconstruction_fixture.yml"
    )
    args.max_epochs = 2
    args.limit_train_batches = 1
    args.limit_val_batches = 1
    args.early_stopping_patience = 0
    args.batch_size = 2
    args.num_workers = 0
    args.wandb = False
    args.output_dir = args.output_dir or os.path.join(
        EXAMPLE_OUTPUT_DIR, "icemix_joint_reconstruction", "smoke_test"
    )


def build_parser() -> ArgumentParser:
    """Return the training argument parser."""
    parser = ArgumentParser(
        description="Train IceMix for joint direction + vertex reconstruction."
    )
    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 80),
        ("early-stopping-patience", 0),
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
    model_group.add_argument(
        "--model-config",
        default=os.path.join(
            GRAPHNET_ROOT_DIR, "configs/models/icemix_joint_reconstruction.yml"
        ),
    )
    model_group.add_argument(
        "--save-model-config",
        default=None,
        help="Write the model config and exit.",
    )

    optimisation = parser.add_argument_group("optimisation")
    for name in MODEL_TUNING_ARGUMENTS:
        optimisation.add_argument(
            "--" + name.replace("_", "-"),
            type=int if name == "scheduler_patience" else float,
            default=None,
            help="Override the saved config (otherwise use model defaults).",
        )
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply z-axis rotation to training batches.",
    )

    run = parser.add_argument_group("run")
    run.add_argument("--run-name", default="icemix")
    run.add_argument("--output-dir", default=None)
    run.add_argument("--resume-from", default=None)
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
        help="Use the small dataset fixture and two short epochs with the full model.",
    )

    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.smoke_test:
        _apply_smoke_test_overrides(parsed)
    main(parsed)
