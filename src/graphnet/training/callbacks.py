"""Callback class(es) for using during model training."""

import logging
import os
from typing import Dict, List, Tuple, TYPE_CHECKING, Any, Optional
import warnings

import numpy as np
import torch
from torch import Tensor
from tqdm.std import Bar

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    Callback,
    TQDMProgressBar,
    EarlyStopping,
)
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.aggregation import CatMetric

from graphnet.utilities.logging import Logger

if TYPE_CHECKING:
    from graphnet.models import Model
    import pytorch_lightning as pl

    # Imported lazily at runtime: `graphnet.models.easy_model` imports this
    # module, so a module-level import of `graphnet.training.loss_functions`
    # (which imports `graphnet.models.model`) would close an import cycle.
    from graphnet.training.loss_functions import PerEventRecordingLoss


class PiecewiseLinearLR(_LRScheduler):
    """Interpolate learning rate linearly between milestones."""

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        factors: List[float],
        last_epoch: int = -1,
    ):
        """Construct `PiecewiseLinearLR`.

        For each milestone, denoting a specified number of steps, a factor
        multiplying the base learning rate is specified. For steps between two
        milestones, the learning rate is interpolated linearly between the two
        closest milestones. For steps before the first milestone, the factor
        for the first milestone is used; vice versa for steps after the last
        milestone.

        Args:
            optimizer: Wrapped optimizer.
            milestones: List of step indices. Must be increasing.
            factors: List of multiplicative factors. Must be same length as
                `milestones`.
            last_epoch: The index of the last epoch.
        """
        # Check(s)
        if milestones != sorted(milestones):
            raise ValueError("Milestones must be increasing")
        if len(milestones) != len(factors):
            raise ValueError(
                "Only multiplicative factor must be specified"
                " for each milestone."
            )

        self.milestones = milestones
        self.factors = factors
        super().__init__(optimizer, last_epoch)

    def _get_factor(self) -> np.ndarray:
        # Linearly interpolate multiplicative factor between milestones.
        return np.interp(self.last_epoch, self.milestones, self.factors)

    def get_lr(self) -> List[float]:
        """Get effective learning rate(s) for each optimizer."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        return [base_lr * self._get_factor() for base_lr in self.base_lrs]


class ProgressBar(TQDMProgressBar):
    """Custom progress bar for graphnet.

    Customises the default progress in pytorch-lightning.
    """

    def _common_config(self, bar: Bar) -> Bar:
        bar.unit = " batch(es)"
        bar.colour = "green"
        return bar

    def init_validation_tqdm(self) -> Bar:
        """Override for customisation."""
        bar = super().init_validation_tqdm()
        bar = self._common_config(bar)
        return bar

    def init_predict_tqdm(self) -> Bar:
        """Override for customisation."""
        bar = super().init_predict_tqdm()
        bar = self._common_config(bar)
        return bar

    def init_test_tqdm(self) -> Bar:
        """Override for customisation."""
        bar = super().init_test_tqdm()
        bar = self._common_config(bar)
        return bar

    def init_train_tqdm(self) -> Bar:
        """Override for customisation."""
        bar = super().init_train_tqdm()
        bar = self._common_config(bar)
        return bar

    def get_metrics(self, trainer: Trainer, model: LightningModule) -> Dict:
        """Override to not show the version number in the logging."""
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def on_train_epoch_start(
        self, trainer: Trainer, model: LightningModule
    ) -> None:
        """Print the results of the previous epoch on a separate line.

        This allows the user to see the losses/metrics for previous
        epochs while the current is training. The default behaviour in
        pytorch- lightning is to overwrite the progress bar from
        previous epochs.
        """
        if trainer.current_epoch > 0:
            self.train_progress_bar.set_postfix(
                self.get_metrics(trainer, model)
            )
            print("")
        super().on_train_epoch_start(trainer, model)
        self.train_progress_bar.set_description(
            f"Epoch {trainer.current_epoch:2d}"
        )

    def on_train_epoch_end(
        self, trainer: Trainer, model: LightningModule
    ) -> None:
        """Log the final progress bar for the epoch to file.

        Don't duplciate to stdout.
        """
        super().on_train_epoch_end(trainer, model)

        if rank_zero_only.rank == 0:
            # Construct Logger
            logger = Logger()

            # Log only to file, not stream
            h = logger.handlers[0]
            assert isinstance(h, logging.StreamHandler)
            level = h.level
            h.setLevel(logging.ERROR)
            logger.info(str(super().train_progress_bar))
            h.setLevel(level)


class GraphnetEarlyStopping(EarlyStopping):
    """Early stopping callback for graphnet."""

    def __init__(self, save_dir: str, **kwargs: Dict[str, Any]) -> None:
        """Construct `GraphnetEarlyStopping` Callback.

        Args:
            save_dir: Path to directory to save best model and config.
            **kwargs: Keyword arguments to pass to `EarlyStopping`. See
                `pytorch_lightning.callbacks.EarlyStopping` for details.
        """
        self.save_dir = save_dir
        super().__init__(**kwargs)

    def setup(
        self,
        trainer: "pl.Trainer",
        graphnet_model: "Model",
        stage: Optional[str] = None,
    ) -> None:
        """Call at setup stage of training.

        Args:
            trainer: The trainer.
            graphnet_model: The model.
            stage: The stage of training.
        """
        super().setup(trainer, graphnet_model, stage)
        os.makedirs(self.save_dir, exist_ok=True)
        graphnet_model.save_config(os.path.join(self.save_dir, "config.yml"))

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", graphnet_model: "Model"
    ) -> None:
        """Call after each train epoch.

        Args:
            trainer: Trainer object.
            graphnet_model: Graphnet Model.

        Returns: None.
        """
        if not self._check_on_train_epoch_end or self._should_skip_check(
            trainer
        ):
            return
        current_best = self.best_score
        self._run_early_stopping_check(trainer)
        if self.best_score != current_best:
            graphnet_model.save_state_dict(
                os.path.join(self.save_dir, "best_model.pth")
            )

    def on_validation_end(
        self, trainer: "pl.Trainer", graphnet_model: "Model"
    ) -> None:
        """Call after each validation epoch.

        Args:
            trainer: Trainer object.
            graphnet_model: Graphnet Model.

        Returns: None.
        """
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        current_best = self.best_score
        self._run_early_stopping_check(trainer)
        if self.best_score != current_best:
            graphnet_model.save_state_dict(
                os.path.join(self.save_dir, "best_model.pth")
            )

    def on_fit_end(
        self, trainer: "pl.Trainer", graphnet_model: "Model"
    ) -> None:
        """Call at the end of training.

        Args:
            trainer: Trainer object.
            graphnet_model: Graphnet Model.

        Returns: None.
        """
        graphnet_model.load_state_dict(
            os.path.join(self.save_dir, "best_model.pth")
        )


class RandomRotationCallback(Callback):
    """Randomly rotate each event about a configurable z-axis pivot."""

    def __init__(
        self,
        seed: Optional[int] = None,
        x_index: int = 0,
        y_index: int = 1,
        direction_key: str = "direction",
        position_keys: Tuple[str, str] = ("position_x", "position_y"),
        azimuth_key: str = "azimuth",
        node_xy_origin: Tuple[float, float] = (0.0, 0.0),
        position_xy_origin: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Construct `RandomRotationCallback`."""
        super().__init__()
        self._x_index = x_index
        self._y_index = y_index
        self._direction_key = direction_key
        self._position_keys = position_keys
        self._azimuth_key = azimuth_key
        self._node_xy_origin = node_xy_origin
        self._position_xy_origin = position_xy_origin

        self._generator = torch.Generator(device="cpu")
        if seed is None:
            self._generator.seed()
        else:
            self._generator.manual_seed(seed)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Rotate `batch` in place before it is passed to the model."""
        for data in batch if isinstance(batch, list) else [batch]:
            self._apply_rotation(data)

    def _apply_rotation(self, data: Any) -> None:
        n_events = data.num_graphs

        # Drawn on CPU so that the augmentation is reproducible from `seed`
        # regardless of which device the batch happens to live on.
        angles = (
            torch.rand(
                n_events, generator=self._generator, dtype=torch.float32
            )
            * 2.0
            * np.pi
        ).to(data.x.device)
        cos_event, sin_event = torch.cos(angles), torch.sin(angles)

        cos_node = cos_event[data.batch].to(data.x.dtype)
        sin_node = sin_event[data.batch].to(data.x.dtype)
        self._rotate_pair(
            data.x,
            self._x_index,
            self._y_index,
            cos_node,
            sin_node,
            self._node_xy_origin,
        )

        direction = getattr(data, self._direction_key, None)
        if direction is not None:
            self._rotate_pair(
                direction.view(n_events, -1),
                0,
                1,
                cos_event.to(direction.dtype),
                sin_event.to(direction.dtype),
            )

        x_key, y_key = self._position_keys
        pos_x = getattr(data, x_key, None)
        pos_y = getattr(data, y_key, None)
        if pos_x is not None and pos_y is not None:
            cos_pos = cos_event.to(pos_x.dtype)
            sin_pos = sin_event.to(pos_x.dtype)
            origin_x, origin_y = self._position_xy_origin
            x = pos_x - origin_x
            y = pos_y - origin_y
            rotated_x = x * cos_pos - y * sin_pos + origin_x
            rotated_y = x * sin_pos + y * cos_pos + origin_y
            setattr(data, x_key, rotated_x)
            setattr(data, y_key, rotated_y)

        azimuth = getattr(data, self._azimuth_key, None)
        if azimuth is not None:
            setattr(
                data,
                self._azimuth_key,
                (azimuth + angles.to(azimuth.dtype)) % (2.0 * np.pi),
            )

    @staticmethod
    def _rotate_pair(
        tensor: Tensor,
        x_index: int,
        y_index: int,
        cos: Tensor,
        sin: Tensor,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Rotate columns `x_index`/`y_index` of `tensor` about `origin`."""
        origin_x, origin_y = origin
        x = tensor[:, x_index].clone() - origin_x
        y = tensor[:, y_index].clone() - origin_y
        tensor[:, x_index] = x * cos - y * sin + origin_x
        tensor[:, y_index] = x * sin + y * cos + origin_y


class PhysicsMetricsCallback(Callback):
    """Log per-event reconstruction quality alongside the training loss."""

    QUANTILES = {"p10": 0.1, "median": 0.5, "p90": 0.9}

    def __init__(
        self,
        position_loss_name: str = "position",
        direction_loss_name: str = "direction",
        sel_energy_min: Optional[float] = 10.0,
    ) -> None:
        """Construct `PhysicsMetricsCallback`.

        Args:
            position_loss_name: `metric_name` of the position recorder.
            direction_loss_name: `metric_name` of the direction recorder.
            sel_energy_min: Energy threshold of `angular_error_deg_sel`, or
                `None` to skip it.
        """
        super().__init__()
        self._position_loss_name = position_loss_name
        self._direction_loss_name = direction_loss_name
        self._sel_energy_min = sel_energy_min
        keys = (
            "position_error_m",
            "angular_error_deg",
            "angular_error_deg_sel",
            "kappa",
            "vmf_calibration",
        )
        self._metrics = {
            phase: {key: CatMetric(nan_strategy="disable") for key in keys}
            for phase in ("train", "val")
        }

    @staticmethod
    def _recorders(
        pl_module: LightningModule,
    ) -> Dict[str, "PerEventRecordingLoss"]:
        """Collect every named `PerEventRecordingLoss` under `pl_module`."""
        from graphnet.training.loss_functions import PerEventRecordingLoss

        return {
            module.metric_name: module
            for module in pl_module.modules()
            if isinstance(module, PerEventRecordingLoss)
            and module.metric_name is not None
        }

    def _per_event_values(
        self, pl_module: LightningModule, batch: Any
    ) -> Dict[str, Tensor]:
        recorders = self._recorders(pl_module)
        values: Dict[str, Tensor] = {}

        position = recorders.get(self._position_loss_name)
        if (
            position is not None
            and position.last_prediction is not None
            and position.last_target is not None
        ):
            prediction = position.last_prediction
            target = position.last_target.reshape(prediction.size(0), -1)
            values["position_error_m"] = torch.linalg.vector_norm(
                prediction[:, :3] - target[:, :3], dim=1
            )

        direction = recorders.get(self._direction_loss_name)
        if (
            direction is not None
            and direction.last_prediction is not None
            and direction.last_target is not None
        ):
            prediction = direction.last_prediction
            target = direction.last_target.reshape(prediction.size(0), -1)
            kappa = prediction[:, 3]
            cosine = (prediction[:, :3] * target[:, :3]).sum(dim=1)
            angle = torch.arccos(cosine.clamp(-1.0, 1.0))
            values["angular_error_deg"] = torch.rad2deg(angle)
            values["kappa"] = kappa
            values["vmf_calibration"] = angle * kappa.clamp_min(0.0).sqrt()

            energy = getattr(batch, "energy", None)
            if self._sel_energy_min is not None and energy is not None:
                selected = energy >= self._sel_energy_min
                values["angular_error_deg_sel"] = values["angular_error_deg"][
                    selected
                ]

        return values

    def _update(
        self, pl_module: LightningModule, phase: str, batch: Any
    ) -> None:
        for key, value in self._per_event_values(pl_module, batch).items():
            metric = self._metrics[phase][key]
            if metric.device != pl_module.device:
                metric.to(pl_module.device)
            metric.update(value)

    def _flush(self, pl_module: LightningModule, phase: str) -> None:
        quantiles = torch.tensor(list(self.QUANTILES.values()))
        logged: Dict[str, Tensor] = {}
        for key, metric in self._metrics[phase].items():
            if metric.update_count == 0:
                continue
            values = metric.compute()
            metric.reset()
            if not isinstance(values, Tensor) or values.numel() == 0:
                # Keep monitored keys present when a selection is empty.
                values = torch.tensor([float("nan")], device=pl_module.device)
            name = f"{phase}/{key}"
            logged[f"{name}_mean"] = values.mean()
            for suffix, value in zip(
                self.QUANTILES,
                torch.quantile(values, quantiles.to(values.device)),
            ):
                logged[f"{name}_{suffix}"] = value
        # `CatMetric.compute` already gathers the values of every rank.
        pl_module.log_dict(logged, sync_dist=False)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Accumulate the training metrics of this batch."""
        self._update(pl_module, "train", batch)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate the validation metrics of this batch."""
        self._update(pl_module, "val", batch)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log and reset the training metrics."""
        self._flush(pl_module, "train")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log and reset the validation metrics."""
        self._flush(pl_module, "val")
