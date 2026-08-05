"""Measure how IceMix degrades as input pulses are removed.

Sweeps the fraction of dropped pulses at inference time and writes one
predictions CSV per setting, so reconstruction quality can be plotted against
how much of the event the model was allowed to see.

`DeepIce` takes an explicit `eval_token_drop` argument, so eval-time dropping
is declared rather than patched onto the backbone after construction.
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.models import StandardModel
from graphnet.models.gnn import DeepIce
from graphnet.training.labels import Direction
from graphnet.data.utilities.paths import resolve_dataset_paths
from graphnet.data.utilities.splits import seeded_event_splits
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import DatasetConfig, ModelConfig
from graphnet.utilities.logging import Logger

ADDITIONAL_ATTRIBUTES = [
    "position_x",
    "position_y",
    "position_z",
    "azimuth",
    "zenith",
    "energy",
    "pid",
    "interaction_type",
    "event_no",
]


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the checkpoint with the lowest validation loss.

    Matches the `best-epoch=..-val_loss=..ckpt` filenames written by the
    training entry point's `ModelCheckpoint`.
    """
    import glob
    import re

    best_path, best_loss = None, float("inf")
    pattern = os.path.join(checkpoint_dir, "best-epoch=*-val_loss=*.ckpt")
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        match = re.search(r"val_loss=([0-9]+\.?[0-9]*)", name)
        if match and float(match.group(1)) < best_loss:
            best_path, best_loss = path, float(match.group(1))
    return best_path


def build_dataloader(
    dataset_config: DatasetConfig,
    split: str,
    batch_size: int,
    num_workers: int,
    split_ratio: List[float],
) -> DataLoader:
    """Return a non-shuffling dataloader over one split of the dataset."""
    paths = resolve_dataset_paths(dataset_config.path)

    representation = dataset_config.data_representation
    if isinstance(representation, dict):
        from graphnet.data.dataset.dataset import parse_data_representation

        representation = parse_data_representation(representation)

    splits = seeded_event_splits(
        paths=paths,
        seed=dataset_config.seed if dataset_config.seed is not None else 42,
        ratio=split_ratio,
        index_column=dataset_config.index_column,
        truth_table=dataset_config.truth_table,
    )
    datasets = [
        SQLiteDataset(
            path=path,
            selection=selection,
            pulsemaps=dataset_config.pulsemaps,
            features=dataset_config.features,
            truth=dataset_config.truth,
            truth_table=dataset_config.truth_table,
            index_column=dataset_config.index_column,
            data_representation=representation,
            labels={
                "direction": Direction(
                    azimuth_key="azimuth", zenith_key="zenith"
                )
            },
        )
        for path, selection in zip(paths, splits[split])
        if selection
    ]
    return DataLoader(
        EnsembleDataset(datasets),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


def main(args: Any) -> None:
    """Run the sweep and write one CSV per drop fraction."""
    logger = Logger()

    model = StandardModel.from_config(
        ModelConfig.load(args.model_config), trust=True
    )
    assert isinstance(model, StandardModel)

    checkpoint = args.checkpoint or find_best_checkpoint(args.checkpoint_dir)
    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint given and none found in {args.checkpoint_dir}"
        )
    logger.info(f"Loading weights from {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state))

    backbone = model.backbone
    if not isinstance(backbone, DeepIce):
        raise TypeError(
            "Token-drop robustness only applies to a `DeepIce` backbone. "
            f"Got {type(backbone).__name__}."
        )

    dataset_config = DatasetConfig.load(args.dataset_config)
    dataloader = build_dataloader(
        dataset_config=dataset_config,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_ratio=args.split_ratio,
    )
    logger.info(f"Evaluating on {len(dataloader.dataset)} events.")

    os.makedirs(args.output_dir, exist_ok=True)
    summary: List[Dict[str, float]] = []

    # Restored afterwards so the loaded model is left as it was found.
    original = (
        backbone.token_drop,
        backbone.drop_chance,
        backbone.eval_token_drop,
    )
    try:
        for fraction in args.drop_fractions:
            backbone.token_drop = fraction
            backbone.drop_chance = 1.0
            backbone.eval_token_drop = fraction > 0.0
            logger.info(f"--- token_drop = {fraction} ---")

            results = model.predict_as_dataframe(
                dataloader,
                additional_attributes=ADDITIONAL_ATTRIBUTES,
                gpus=args.gpus,
            )
            results["token_drop"] = fraction
            path = os.path.join(
                args.output_dir, f"token_drop_{fraction:.3f}.csv"
            )
            results.to_csv(path, index=False)
            logger.info(f"Wrote {path}")
            summary.append({"token_drop": fraction, "n_events": len(results)})
    finally:
        (
            backbone.token_drop,
            backbone.drop_chance,
            backbone.eval_token_drop,
        ) = original

    pd.DataFrame(summary).to_csv(
        os.path.join(args.output_dir, "sweep_summary.csv"), index=False
    )
    logger.info(f"Sweep complete: {args.output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Sweep inference-time pulse dropping for a trained IceMix."
    )
    parser.with_standard_arguments(
        "gpus", ("batch-size", 256), ("num-workers", 3)
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument(
        "--dataset-config",
        default="configs/datasets/icecube86_joint_reconstruction.yml",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "validation", "test"]
    )
    parser.add_argument(
        "--split-ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1]
    )
    parser.add_argument(
        "--drop-fractions",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Pulse-drop fractions to evaluate (default: %(default)s)",
    )
    parser.add_argument("--output-dir", default="outputs/token_drop_sweep")

    args, _ = parser.parse_known_args()
    main(args)
