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

from graphnet.utilities.logging import Logger

if TYPE_CHECKING:
    from graphnet.models import Model
    import pytorch_lightning as pl


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
    """Randomly rotate each event about the detector's z-axis.

    Augments training batches in place: sensor `(x, y)` coordinates and the
    corresponding truth quantities are rotated by a per-event angle drawn
    uniformly from `[0, 2 * pi)`. Because IceCube's geometry is (to good
    approximation) azimuthally symmetric, this maps a physical event onto
    another equally physical event.

    The rotation is applied to the *standardized* node features. This is exact
    as long as the detector applies the same offset-free scaling to `x` and
    `y`, which holds for `IceCube86` (`dom_x`, `dom_y` and `dom_z` all map
    through `x / 500`). Verify this before using the callback with another
    `Detector`.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        x_index: int = 0,
        y_index: int = 1,
        direction_key: str = "direction",
        position_keys: Tuple[str, str] = ("position_x", "position_y"),
        azimuth_key: str = "azimuth",
    ) -> None:
        """Construct `RandomRotationCallback`.

        Args:
            seed: Seed of the angle generator. If None, the generator is seeded
                non-deterministically.
            x_index: Column of `Data.x` holding the sensor x-coordinate.
            y_index: Column of `Data.x` holding the sensor y-coordinate.
            direction_key: Attribute holding the truth direction unit vector of
                shape `[batch_size, 3]`. Skipped when absent.
            position_keys: Attributes holding the truth vertex x- and
                y-coordinates. Skipped when absent.
            azimuth_key: Attribute holding the truth azimuth. Skipped when
                absent.
        """
        super().__init__()
        self._x_index = x_index
        self._y_index = y_index
        self._direction_key = direction_key
        self._position_keys = position_keys
        self._azimuth_key = azimuth_key

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
        n_events = int(data.batch.max().item()) + 1

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
            data.x, self._x_index, self._y_index, cos_node, sin_node
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
            rotated_x = pos_x * cos_pos - pos_y * sin_pos
            rotated_y = pos_x * sin_pos + pos_y * cos_pos
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
    ) -> None:
        """Rotate columns `x_index`/`y_index` of `tensor` in place."""
        x = tensor[:, x_index].clone()
        y = tensor[:, y_index].clone()
        tensor[:, x_index] = x * cos - y * sin
        tensor[:, y_index] = x * sin + y * cos
