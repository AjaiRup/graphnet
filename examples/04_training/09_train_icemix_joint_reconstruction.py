"""Train IceMix for joint neutrino direction and vertex reconstruction.

Two prediction heads share one `DeepIce` backbone: a vertex head trained with a
Euclidean distance loss and a direction head trained with a 3D von Mises-Fisher
loss, combined as `alpha * vertex + direction`.

Defaults follow `graphnet-cursor`'s `ice_mix/conf/` rather than upstream's, so
this reproduces the setup those runs used.
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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
    PerEventRecordingLoss,
    VonMisesFisher3DLoss,
)
from graphnet.training.utils import collator_sequence_buckleting
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import DatasetConfig, ModelConfig
from graphnet.utilities.logging import Logger
from graphnet.data.utilities.lengths import event_lengths
from graphnet.data.utilities.paths import resolve_dataset_paths
from graphnet.data.utilities.splits import seeded_event_splits

POSITION_LABELS = ["position_x", "position_y", "position_z"]

# Metres per unit of the standardized position targets; also handed to
# `PhysicsMetricsCallback` so `position_error_m` stays in metres.
POSITION_SCALE_M = 100.0


def scale_position(x):  # type: ignore[no-untyped-def]
    """Map vertex targets from metres to detector-centred O(1) units.

    Geometry constants only, mirroring the `IceCubeDeepCore` feature map
    (`detector/icecube.py`): x/y centred on the DeepCore string-36 axis
    (46.3, -34.9), z shifted by the same +350 m upstream applies to input z,
    everything divided by 100 m. Centring is the operative part: with an
    uncentred target, any train/eval calibration drift of the backbone
    becomes a systematic vertex offset proportional to the target mean --
    measured at ~+20 m in z on `icemix-11666796`, whose head scaled by a
    bare x100 without centring.

    Constants are inlined, and the signature is unannotated, so that the
    function survives serialisation into a model config. Annotating it is
    not an option: `ModelConfig.FUNCTION_DEFINITION_PATTERN` matches a def
    line ending `):`, so a return annotation of any kind -- including a
    quoted one -- stops the config round-tripping. Hence the suppression
    rather than a type.
    """
    x[:, 0] = (x[:, 0] - 46.3) / 100.0
    x[:, 1] = (x[:, 1] + 34.9) / 100.0
    x[:, 2] = (x[:, 2] + 350.0) / 100.0
    return x


def unscale_position(x):  # type: ignore[no-untyped-def]
    """Invert `scale_position`, recovering metres at inference time."""
    x[:, 0] = x[:, 0] * 100.0 + 46.3
    x[:, 1] = x[:, 1] * 100.0 - 34.9
    x[:, 2] = x[:, 2] * 100.0 - 350.0
    return x


def single_threaded_worker(worker_id: int) -> None:
    """Pin a dataloader worker to one compute thread.

    Workers inherit `OMP_NUM_THREADS` from the launcher, which sets 4 so the
    main process can use them. Multiplied out that is 8 ranks x N workers x 4
    threads against the allocation's cores -- 128 threads on 32 cores at 4
    workers per rank. The per-event work is a SQLite read and a handful of
    tiny tensor ops, none of which parallelise, so the extra threads only
    contend.
    """
    torch.set_num_threads(1)


def batch_limit(value: str) -> Any:
    """Parse a `--limit-*-batches` value as Lightning interprets it.

    Lightning reads an `int` as a number of batches and a `float` as a
    *fraction* of the epoch, rejecting floats above 1.0. Plain `type=float`
    would therefore turn `--limit-train-batches 50` into "5000% of the epoch"
    and raise. Keep the distinction the user wrote.
    """
    return float(value) if "." in value else int(value)


def build_data_representation(
    features: List[str],
    max_pulses: int,
    hlc_name: Optional[str],
    add_ice_properties: bool,
    add_static_features: bool = False,
) -> GraphDefinition:
    """Return the pulse-sequence representation IceMix consumes.

    `GraphDefinition` rather than `KNNGraph`: `DeepIce` never reads
    `edge_index`, so building a KNN graph would cost time for nothing.

    `add_static_features` defaults to False for the same reason. It copies
    every column of `data.x` back out as its own per-node attribute, which
    doubles the per-pulse tensor volume through collate, pinning and the
    host-to-device copy -- and `DeepIce` reads only `data.x` and `data.batch`.
    Measured at 5.53 -> 7.83 batch/s on the training dataloader.
    """
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
    """Return this process' `(rank, world_size)` before DDP has started.

    `LengthGroupedSampler` shards the epoch itself, so it has to know the rank
    at construction time -- which is before `Trainer.fit`, and therefore before
    any process group exists. Under `srun --ntasks-per-node=N` that is free:
    SLURM sets `SLURM_PROCID`/`SLURM_NTASKS` in every task before Python
    starts, and Lightning's `SLURMEnvironment` reads exactly the same two
    variables, so the sampler and the Trainer cannot disagree.

    Lightning's own subprocess launcher cannot support this. It forks the
    children from *this* process inside `Trainer.fit`
    (`strategies/launchers/subprocess_script.py`), long after the dataloaders
    were built, so the parent resolves to `(0, 1)` and hands itself the whole
    epoch while its children take an eighth each -- a deadlock on the first
    collective after they run out. Refuse to guess rather than paper over it.
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
    add_default_labels: bool = True,
    group_factor: int = 0,
    length_cache_dir: Optional[str] = None,
    max_pulses: Optional[int] = None,
    seed: int = 42,
    collator_splits: Optional[List[float]] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, DataLoader]:
    """Return the train/validation/test dataloaders.

    Two paths, chosen by whether the dataset config carries a `selection`:

    * With a selection, upstream's `DataLoader.from_dataset_config` does
      everything. Query strings are evaluated against each database
      separately, so they behave correctly even though `event_no` is not
      unique across the flavour samples.
    * Without one, the split is resolved by `seeded_event_splits`, which
      reproduces the physics team's shuffle-with-a-fixed-seed partition. That
      cannot be expressed as a `DatasetConfig` selection because it needs
      per-database index lists, so the datasets are composed here from
      upstream primitives instead.

    Every batch is padded to its longest event, and with events drawn at
    random that is almost always the `max_pulses` cap: over the full 10.4 M
    sample the mean padded length is 252 to carry 37 real pulses, so ~85% of
    the tokens - and ~98% of the `L**2` terms the relative-position encoder
    builds - are padding. There are two ways to stop paying for that, and they
    are not close:

    * `group_factor > 0` (the default) orders events *by* pulse count via
      `LengthGroupedSampler`, so consecutive indices - and therefore each
      batch - are length-matched. Mean padded length 46 at `group_factor=16`,
      against 252 unordered: **9.5 min/epoch measured, against 18.3 for the
      previous bucketing scheme and 47.3 for none**. Nothing is dropped -
      `DistributedSampler` pads the index list rather than truncating it.

      It is not free. DDP averages gradients, so an optimiser step sees
      `batch_size * world_size` events, and grouping tightly enough to remove
      the padding necessarily makes that sample length-homogeneous: the
      standard deviation of event length within one step is 40.6 unordered,
      31.8 under the old scheme, 4.9 here. Per epoch this run tracks ~0.1-0.2
      above `outputs/acc-nobucket` on `val_loss` and is noisier; per minute it
      is ahead of both. `--accumulate-grad-batches` buys the diversity back at
      no throughput cost, because consecutive steps are at unrelated lengths.
    * `collator_splits` sorts each batch by pulse count and cuts it into
      sub-batches, each padded to its own longest event.
      `StandardModel.forward` runs the backbone per sub-batch and concatenates
      the embeddings before the loss; the backbone is per-event, so the
      gradient is *identical* to not splitting. Nothing is dropped and the
      result cannot move. It is also **worth almost nothing: 685 ms/step
      measured**, ~7% over no strategy at all.

    That gap is structural, not a tuning problem, and a token-cost model does
    not predict it. Sorting a *random* batch of 256 leaves its top slice at
    the cap, so every step still pays one full-length pass - and at these
    shapes the model is latency-bound rather than throughput-bound, so a
    26-event pass at L=256 costs nearly what a 256-event one does (measured
    668 ms). Length grouping is the only scheme here that removes the
    full-length pass from the ~99% of steps that do not need one; no amount of
    within-batch reorganisation reproduces that. `--group-factor 0` restores
    random batching.

    Grouping happens across the *whole* split and is sharded afterwards, so
    every rank reads the same window of the same sorted block and a DDP step
    - which costs what its slowest rank costs - is padded to one length rather
    than to the maximum over `world_size` independent draws. Getting that
    backwards is expensive and silent: it looks correct on every rank
    individually and costs 115 padded tokens per step to carry 44.
    """
    logger = Logger()
    collate = (
        collator_sequence_buckleting(list(collator_splits))
        if collator_splits
        else None
    )
    if collate is not None and group_factor > 0:
        # Both would work, but stacking them means length-matched batches get
        # cut again by fixed *fractions*, which mostly adds backbone calls.
        logger.warning(
            "Both --group-factor and --collator-splits are set. Length "
            "grouping wins; the collator is skipped."
        )
        collate = None

    if dataset_config.selection is not None:
        if group_factor > 0:
            logger.warning(
                "Length grouping needs the per-database index lists that "
                "`seeded_event_splits` produces, and the dataset config "
                "carries a `selection`. Falling back to random batching."
            )
        if collate is not None:
            # `from_dataset_config` builds every split from one set of kwargs,
            # and `predict_as_dataframe` reads `batch[attr]`, which a
            # list-of-`Batch` does not support -- so applying the collator
            # here would break the `test` loader.
            logger.warning(
                "The sequence collator is skipped when the dataset config "
                "carries a `selection`: every split would share it, including "
                "`test`, which `predict_as_dataframe` cannot read."
            )
        logger.info("Resolving splits from the dataset config's `selection`.")
        dataloaders = DataLoader.from_dataset_config(
            dataset_config, **dataloader_kwargs
        )
        assert isinstance(dataloaders, dict)
        return dataloaders

    # Keeps the shipped config free of machine-specific absolute paths.
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

    # `Dict[str, Any]`, not the inferred union: splatting a heterogeneous
    # dict into a typed signature is uncheckable, and the narrow inferred
    # type makes mypy reject every keyword rather than none.
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
        add_default_labels=add_default_labels,
    )

    dataloaders = {}
    for name, per_database in splits.items():
        # Track which paths actually contributed: a database with an empty
        # selection is skipped, and `EnsembleDataset` concatenates only the
        # datasets it was given. The pulse-count array has to line up with
        # that concatenation, not with `paths`.
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

        # `test` is left on the default sampler, and not because grouping
        # would cost it anything -- it drops nothing. `predict_as_dataframe`
        # re-iterates the loader in the main process to collect
        # `additional_attributes`, and `EasySyntax.predict` refuses to do that
        # unless `dataloader.sampler` is a `SequentialSampler`, because any
        # other order would pair each row with the wrong event's truth.
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
                    # Validation batches are fixed, so `val_loss` compares
                    # like with like across epochs; training reshuffles.
                    shuffle=do_shuffle(name),
                    seed=seed,
                    group_factor=group_factor,
                ),
                # Drop the epoch's remainder when training. `num_samples` is
                # rarely a multiple of `batch_size` -- at 8.3 M events over 8
                # ranks it leaves 35 per rank, so the last optimiser step of
                # every epoch is taken on 280 events against 2048, a ~7x
                # noisier gradient immediately before validation runs. That
                # costs 280 events an epoch, 0.003%, against the 2904 the
                # previous bucketing scheme dropped. Validation keeps its
                # remainder: `val_loss` should be over the whole split.
                drop_last=do_shuffle(name),
                **dataloader_kwargs,
            )
            continue

        # `test` keeps plain collation whichever scheme is on: the collator
        # hands the model a list of `Batch`, and `predict_as_dataframe`
        # reads `batch[attr]` off a single one.
        extra: Dict[str, Any] = {}
        if collate is not None and name != "test":
            logger.info(
                f"Sequence-collating the '{name}' split at "
                f"{list(collator_splits or [])}."
            )
            extra["collate_fn"] = collate

        dataloaders[name] = DataLoader(
            EnsembleDataset(datasets),
            shuffle=do_shuffle(name),
            **dataloader_kwargs,
            **extra,
        )
    return dataloaders


def build_model(args: Any, n_features: int) -> StandardModel:
    """Return the `StandardModel` to train.

    Either loaded from a `ModelConfig` or assembled from the command-line
    arguments. The assembled model is the source of truth for the shipped
    configs - regenerate them with `--save-model-config` rather than editing
    the YAML by hand.
    """
    data_representation = build_data_representation(
        features=args.features,
        max_pulses=args.max_pulses,
        hlc_name=args.hlc_name or None,
        add_ice_properties=args.add_ice_properties,
        add_static_features=args.add_static_features,
    )

    if args.model_config:
        Logger().info(f"Loading model from {args.model_config}")
        model = StandardModel.from_config(
            ModelConfig.load(args.model_config), trust=True
        )
        assert isinstance(model, StandardModel)
        return model

    backbone = DeepIce(
        hidden_dim=args.hidden_dim,
        mlp_ratio=args.mlp_ratio,
        seq_length=args.seq_length,
        depth=args.depth,
        head_size=args.head_size,
        depth_rel=args.n_rel,
        n_rel=args.n_rel,
        scaled_emb=args.scaled_emb,
        include_dynedge=False,
        n_features=n_features,
        init_values=args.init_values,
        init_values_rel=args.init_values,
        dropout=args.dropout,
        attn_drop=args.attn_drop,
        proj_drop=args.proj_drop,
        drop_path_rate=args.drop_path_rate,
        token_drop=args.token_drop,
        drop_chance=args.drop_chance,
        relative_encoder=args.relative_encoder,
        n_freq=args.n_freq,
        norm_cls=args.norm_cls,
    )

    tasks = [
        # Upstream `IdentityTask` + transform hooks instead of
        # `PositionReconstruction`: the head predicts in standardized units
        # (see `scale_position`) and the inverse transform recovers metres at
        # inference. This replaces the undocumented in-place x100 of
        # `PositionReconstruction._forward`.
        IdentityTask(
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
                    loss_factors=[args.alpha],
                ),
                metric_name="position",
            ),
            transform_target=scale_position,
            transform_inference=unscale_position,
        ),
        DirectionReconstructionWithKappaDecoupled(
            hidden_size=backbone.nb_outputs,
            target_labels=["direction"],
            loss_function=PerEventRecordingLoss(
                VonMisesFisher3DLoss(
                    kappa_switch=args.kappa_switch,
                    approximation=args.vmf_approximation,
                ),
                metric_name="direction",
            ),
        ),
    ]

    return StandardModel(
        data_representation=data_representation,
        backbone=backbone,
        tasks=tasks,
        optimizer_class=AdamW,
        optimizer_kwargs={"lr": args.lr, "eps": 1e-05},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={
            "mode": "min",
            "patience": args.scheduler_patience,
            "factor": args.scheduler_factor,
        },
        scheduler_config={"monitor": "val_loss", "frequency": 1},
    )


def build_callbacks(args: Any, output_dir: str) -> List[Any]:
    """Return the training callbacks."""
    callbacks: List[Any] = [
        ProgressBar(),
        PhysicsMetricsCallback(position_scale=POSITION_SCALE_M),
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        # `val_loss` is direction-dominated and its scale depends on the
        # position-target units, so the vertex-best epoch is tracked on the
        # physical metric directly (logged into `trainer.callback_metrics`
        # by `PhysicsMetricsCallback`).
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="best-position-{epoch:02d}",
            auto_insert_metric_name=False,
            monitor="val/position_error_m_mean",
            mode="min",
            save_top_k=1,
        ),
    ]
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=args.early_stopping_patience,
                mode="min",
            )
        )
    if args.augment_rotation:
        Logger().info("Enabling on-the-fly random rotation augmentation.")
        callbacks.append(RandomRotationCallback(seed=args.rotation_seed))
    return callbacks


def main(args: Any) -> None:
    """Train the model and write predictions to `--output-dir`."""
    logger = Logger()
    if args.seed is not None:
        from pytorch_lightning import seed_everything

        seed_everything(args.seed, workers=True)

    if args.save_model_config:
        # Dump-and-exit: the shipped `configs/models/*.yml` are generated by
        # `Model.save_config` from these arguments, never hand-written.
        model = build_model(args, n_features=args.n_features)
        model.save_config(args.save_model_config)
        logger.info(f"Wrote model config to {args.save_model_config}")
        return

    dataset_config = DatasetConfig.load(args.dataset_config)
    n_features = args.n_features or len(dataset_config.features)

    output_dir = args.output_dir or os.path.join(
        EXAMPLE_OUTPUT_DIR, "icemix_joint_reconstruction", args.run_name
    )
    if rank_zero_only.rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    data_representation = build_data_representation(
        features=args.features,
        max_pulses=args.max_pulses,
        hlc_name=args.hlc_name or None,
        add_ice_properties=args.add_ice_properties,
        add_static_features=args.add_static_features,
    )

    dataloader_kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        dataloader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
            worker_init_fn=single_threaded_worker,
        )
    else:
        dataloader_kwargs.update(
            persistent_workers=False, prefetch_factor=None
        )

    # Resolved before the dataloaders, and deliberately not inside the
    # sampler: it has to be the same answer the Trainer will reach later.
    rank, world_size = ddp_ranks(len(args.gpus) if args.gpus else 1)

    dataloaders = build_dataloaders(
        dataset_config=dataset_config,
        data_representation=data_representation,
        dataloader_kwargs=dataloader_kwargs,
        split_ratio=args.split_ratio,
        split_cache_dir=args.split_cache_dir or args.cache_dir,
        add_default_labels=args.add_default_labels,
        group_factor=args.group_factor,
        length_cache_dir=args.length_cache_dir or args.cache_dir,
        max_pulses=args.max_pulses,
        seed=args.seed if args.seed is not None else 42,
        collator_splits=args.collator_splits,
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

    model = build_model(args, n_features=n_features)

    if args.fine_tune_from:
        logger.info(f"Loading weights for fine-tuning: {args.fine_tune_from}")
        checkpoint = torch.load(
            args.fine_tune_from, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))

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
        # Record what was actually run. Without this the W&B run carries
        # metrics but no hyperparameters, which makes two runs impossible to
        # tell apart after the fact. Rank zero only - every rank writing the
        # same config races.
        if rank_zero_only.rank == 0:
            wandb_logger.experiment.config.update(
                {**vars(args), "dataset_config_path": args.dataset_config},
                allow_val_change=True,
            )
        loggers.append(wandb_logger)

    # Pin `SLURMEnvironment` rather than letting Lightning detect it.
    #
    # `SLURMEnvironment.detect()` is false whenever SLURM_JOB_NAME is "bash"
    # or "interactive", which is the `salloc` default -- Lightning would then
    # fall back to its own subprocess launcher and fork `len(--gpus)` children
    # inside *each* of the srun tasks. Pinning it makes the submitted and the
    # interactive paths behave identically, and it is the same environment
    # `ddp_ranks` read the sampler's shard from, so the two cannot disagree.
    #
    # Nothing else is needed on the sampler side: `LengthGroupedSampler` is a
    # `DistributedSampler`, so Lightning's `_requires_distributed_sampler` is
    # already false and the dataloader is handed to the strategy untouched.
    # Leaving `use_distributed_sampler` at its default also means
    # `--group-factor 0` still gets a correct sampler injected for free.
    extra_trainer_kwargs: Dict[str, Any] = {}
    if world_size > 1 and args.distribution_strategy != "auto":
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
        callbacks=build_callbacks(args, output_dir),
        ckpt_path=args.resume_from,
        logger=loggers,
        distribution_strategy=args.distribution_strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0,
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
        dataset_config.index_column,
    ]
    results = model.predict_as_dataframe(
        prediction_loader,
        additional_attributes=wanted_attributes,
        gpus=args.gpus,
        # The `test` loader deliberately carries no sampler, so Lightning's
        # default `use_distributed_sampler=True` shards it across the ranks --
        # and `predict_as_dataframe` then walks the *unsharded* loader to
        # collect `additional_attributes`, gets `world_size` times as many
        # rows as predictions, and drops every attribute with a warning. The
        # result is a file holding one rank's eighth of the split, with no
        # truth columns and no `event_no` to join on. Predicting on every rank
        # is redundant, but a correct report is worth one pass.
        use_distributed_sampler=False,
    )

    # `predict_as_dataframe` skips attributes it cannot line up rather than
    # raising, which turns the above into a silently unusable file. Refuse it.
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

    # Rank-guarded for the same reason as `save_state_dict` above: every rank
    # reaches this line and they would all write the same path at once.
    if rank_zero_only.rank == 0:
        predictions_dir = os.path.join(output_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        results.to_csv(
            os.path.join(predictions_dir, "results.csv"), index=False
        )
        logger.info(f"Wrote predictions to {predictions_dir}/results.csv")


def _apply_smoke_test_overrides(args: Any) -> None:
    """Shrink the run so it completes on CPU against the test fixture.

    The fixture holds five events and is schema-identical to the
    production databases, so this exercises the full path end to end in
    seconds.
    """
    args.dataset_config = os.path.join(
        DATASETS_CONFIG_DIR, "icecube86_joint_reconstruction_fixture.yml"
    )
    args.max_epochs = 2
    args.early_stopping_patience = 0
    args.batch_size = 2
    args.num_workers = 0
    args.hidden_dim = 32
    args.seq_length = 16
    args.depth = 2
    args.n_rel = 1
    args.head_size = 8
    args.max_pulses = 32
    args.precision = "32-true"
    args.gpus = None
    args.wandb = False
    args.pin_memory = False
    args.distribution_strategy = "auto"
    args.split_ratio = [0.6, 0.2, 0.2]
    # Length grouping is left ON. The previous bucketing scheme had to be
    # disabled here because it emitted only full batches and the fixture holds
    # five events, so no bucket could reach `batch_size`.
    # `LengthGroupedSampler` drops nothing, and a five-event split exercises
    # it fine -- the only place the grouped path runs on CPU before a GPU job.
    #
    # `--collator-splits` is left alone rather than forced off: it is off by
    # default, and passing it explicitly is the only way the list-of-`Batch`
    # route gets exercised on CPU. It degrades gracefully at batch_size=2 --
    # the fractional cuts collapse to two sub-batches of one event, and empty
    # ones are skipped.
    args.output_dir = args.output_dir or os.path.join(
        EXAMPLE_OUTPUT_DIR, "icemix_joint_reconstruction", "smoke_test"
    )


def build_parser() -> ArgumentParser:
    """Return the argument parser for this entry point.

    A function rather than a block under `__main__` so that a harness needing
    a populated argument namespace -- a benchmark building the same model this
    script does -- can call `build_parser().parse_args([])` instead of
    hand-listing the defaults, which drift every time an argument is added.
    """
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
        "--features",
        nargs="+",
        default=[
            "dom_x",
            "dom_y",
            "dom_z",
            "dom_time",
            "charge",
            "hlc",
        ],
        help=(
            "Pulse features, in order. Must match the dataset config's "
            "`features` column for column. Column 5 is embedded as a binary "
            "flag by `FourierEncoder` (default: %(default)s). "
            "`graphnet-cursor` puts `rde` here instead of `hlc`, which is the "
            "one substantive feature difference between the two forks; see "
            "configs/datasets/icecube86_joint_reconstruction_rde.yml. "
            "`pmt_area` is genuinely unused: it would sit in column 6, past "
            "`--n-features` 6, and never reach a layer."
        ),
    )
    data.add_argument("--n-features", type=int, default=6)
    data.add_argument(
        "--add-static-features",
        action="store_true",
        help=(
            "Copy each column of `data.x` back out as its own per-node "
            "attribute. `DeepIce` reads neither, and it roughly halves "
            "dataloader throughput; only needed by models that consume "
            "`data.features`, e.g. `Coarsening` or `NodeRNN`."
        ),
    )
    data.add_argument(
        "--add-default-labels",
        action="store_true",
        help=(
            "Attach the ten derived classification labels (`muon`, `v_e`, "
            "`track`, ...) that `Dataset._get_labels` builds from the truth "
            "table. This model reads none of them, so they are off by "
            "default here; `Dataset` itself still defaults them on."
        ),
    )
    data.add_argument("--max-pulses", type=int, default=256)
    data.add_argument("--hlc-name", default="hlc")
    data.add_argument("--add-ice-properties", action="store_true")
    data.add_argument("--prefetch-factor", type=int, default=4)
    data.add_argument(
        "--no-pin-memory", dest="pin_memory", action="store_false"
    )
    data.add_argument(
        "--split-ratio",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        help=(
            "Train/validation/test fractions used when the dataset config "
            "has no `selection` (default: %(default)s)"
        ),
    )
    data.add_argument(
        "--collator-splits",
        nargs="*",
        type=float,
        default=[],
        help=(
            "Sort each batch by pulse count and cut it at these fractions, "
            "padding each sub-batch to its own longest event. Gradient-"
            "identical to not splitting, because the backbone is per-event "
            "and the embeddings are concatenated before the loss -- but "
            "measured at 685 ms/step against 730 for no batching strategy at "
            "all, i.e. worth ~7%%, where --group-factor is worth ~7x. Sorting "
            "a random batch of 256 leaves its top slice at the --max-pulses "
            "cap, so every step still pays one full-length pass, and at these "
            "shapes a 26-event pass at L=256 costs nearly what a 256-event "
            "one does. Kept as an opt-in for the record, not because it is "
            "recommended (default: %(default)s)"
        ),
    )
    data.add_argument(
        "--group-factor",
        "--bucket-width",  # kept so queued jobs and older scripts still run
        type=int,
        default=16,
        dest="group_factor",
        help=(
            "Order events by pulse count before batching, via "
            "`LengthGroupedSampler`, in megabatches this many global steps "
            "wide. Random batches pad to the `--max-pulses` cap almost every "
            "time (mean padded length 252 to carry 37 real pulses); grouping "
            "brings that to 58 at 8, 46 at 16 and 39 at 50, against a "
            "no-padding floor of 37. Measured end to end at 9.5 min/epoch "
            "against 18.3 for the old bucketing and 47.3 for none. Nothing is "
            "dropped. It costs accuracy per epoch, and more than the old "
            "bucketing did: DDP averages gradients, so an optimiser step sees "
            "batch_size x world_size events, and the standard deviation of "
            "event length within one step falls from 40.6 unordered - and "
            "31.8 under the old scheme, which drew each rank from an "
            "unrelated bucket - to 8.9 at 8 and 4.9 at 16. The trade still "
            "pays at matched wall clock; lower this to loosen it, or use "
            "--accumulate-grad-batches, which restores the diversity at no "
            "throughput cost. 0 restores random batching (default: "
            "%(default)s)"
        ),
    )
    data.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Default for both --split-cache-dir and --length-cache-dir, for "
            "the common case where one directory holds everything. Either "
            "specific flag still wins where it is given (default: %(default)s)"
        ),
    )
    data.add_argument("--split-cache-dir", default=None)
    data.add_argument(
        "--length-cache-dir",
        default=None,
        help=(
            "Where per-event pulse counts are cached for `--group-factor`. "
            "Without it every rank rescans the pulsemap at start-up, which "
            "is 8 s for numu.db but 39 min for the fragmented nutau.db "
            "(default: %(default)s)"
        ),
    )
    data.add_argument("--augment-rotation", action="store_true")
    data.add_argument("--rotation-seed", type=int, default=None)

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model-config", default=None)
    model_group.add_argument(
        "--save-model-config",
        default=None,
        help=(
            "Write the model's `ModelConfig` to this path and exit without "
            "training. Use this to regenerate configs/models/*.yml."
        ),
    )
    model_group.add_argument("--hidden-dim", type=int, default=384)
    model_group.add_argument("--seq-length", type=int, default=128)
    model_group.add_argument("--depth", type=int, default=12)
    model_group.add_argument("--head-size", type=int, default=32)
    model_group.add_argument("--n-rel", type=int, default=4)
    model_group.add_argument("--mlp-ratio", type=int, default=4)
    model_group.add_argument("--init-values", type=float, default=1.0)
    model_group.add_argument("--n-freq", type=float, default=10000.0)
    model_group.add_argument(
        "--no-scaled-emb", dest="scaled_emb", action="store_false"
    )
    model_group.add_argument(
        "--relative-encoder",
        default="spacetime",
        choices=["spacetime", "mahalanobis"],
    )
    model_group.add_argument("--dropout", type=float, default=0.1)
    model_group.add_argument("--attn-drop", type=float, default=0.05)
    model_group.add_argument("--proj-drop", type=float, default=0.1)
    model_group.add_argument("--drop-path-rate", type=float, default=0.2)
    model_group.add_argument("--token-drop", type=float, default=0.0)
    model_group.add_argument("--drop-chance", type=float, default=1.0)
    model_group.add_argument(
        "--norm-cls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "LayerNorm the CLS token before the task heads, pinning the "
            "readout scale so train/eval dropout differences cannot shift "
            "the vertex. --no-norm-cls restores the pre-fix behaviour of "
            "runs up to icemix-11666796 (default: %(default)s)"
        ),
    )

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--lr", type=float, default=6.25e-5)
    optimisation.add_argument(
        "--alpha",
        type=float,
        default=2.6,
        help=(
            "Weight of the vertex loss term. The vertex loss is a Euclidean "
            "distance over targets standardized to 100 m units, so 2.6 "
            "reproduces the 0.026-per-metre weighting used while the loss "
            "was in raw metres (default: %(default)s)"
        ),
    )
    optimisation.add_argument("--kappa-switch", type=float, default=700.0)
    optimisation.add_argument(
        "--vmf-approximation",
        default="tight",
        choices=["tight", "bound"],
        help=(
            "Asymptotic form of log C_m(kappa) used above --kappa-switch. "
            "'tight' tracks the exact value closely but is not a lower "
            "bound; 'bound' is a lower bound, so the loss is an upper bound "
            "on the negative log-likelihood, but is much less accurate "
            "(default: %(default)s)"
        ),
    )
    optimisation.add_argument("--scheduler-patience", type=int, default=6)
    optimisation.add_argument("--scheduler-factor", type=float, default=0.5)
    optimisation.add_argument("--precision", default="16-mixed")
    optimisation.add_argument("--accumulate-grad-batches", type=int, default=1)
    optimisation.add_argument(
        "--limit-train-batches",
        type=batch_limit,
        default=None,
        help=(
            "Batches per training epoch. An integer is a count, a decimal is "
            "a fraction of the epoch (default: %(default)s)"
        ),
    )
    optimisation.add_argument(
        "--limit-val-batches", type=batch_limit, default=None
    )
    optimisation.add_argument("--distribution-strategy", default="ddp")
    optimisation.add_argument("--seed", type=int, default=42)

    run = parser.add_argument_group("run")
    run.add_argument("--run-name", default="icemix")
    run.add_argument("--output-dir", default=None)
    run.add_argument("--resume-from", default=None)
    run.add_argument("--fine-tune-from", default=None)
    run.add_argument("--wandb", action="store_true")
    run.add_argument("--wandb-project", default="IceMix")
    run.add_argument("--wandb-entity", default=None)
    run.add_argument("--wandb-run-id", default=None)
    run.add_argument(
        "--skip-prediction",
        action="store_true",
        help=(
            "Train, then stop. Inference runs over the whole held-out split "
            "and is not affected by --limit-*-batches, so it dominates the "
            "wall clock of a short run."
        ),
    )
    run.add_argument(
        "--smoke-test",
        action="store_true",
        help="Shrink everything to run on CPU against the test fixture.",
    )

    return parser


if __name__ == "__main__":
    parsed, _ = build_parser().parse_known_args()
    if parsed.smoke_test:
        _apply_smoke_test_overrides(parsed)
    main(parsed)
