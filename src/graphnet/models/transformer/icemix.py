"""Transformer for joint direction and vertex reconstruction."""

import math
from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from graphnet.models import Model, StandardModel
from graphnet.models.components.attention_blocks import Block, Block_rel
from graphnet.models.components.embedding import (
    FourierEncoder,
    SpacetimeEncoderEPJC,
)
from graphnet.models.data_representation import GraphDefinition
from graphnet.models.data_representation.graphs.nodes import IceMixNodes
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.task.reconstruction import (
    DirectionReconstructionWithLogKappa,
    PositionReconstruction,
)
from graphnet.models.utils import array_to_sequence
from graphnet.training.loss_functions import (
    EnsembleLoss,
    EuclideanDistanceLoss,
    PerEventRecordingLoss,
    VonMisesFisher3DLoss,
)
from graphnet.utilities.config import DatasetConfig

FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "rde"]
POSITION_LABELS = ["position_x", "position_y", "position_z"]


def build_data_representation() -> GraphDefinition:
    """Return the data representation the model is trained on."""
    return GraphDefinition(
        detector=IceCube86(),
        node_definition=IceMixNodes(
            input_feature_names=FEATURES,
            max_pulses=256,
            z_name="dom_z",
            hlc_name=None,
            add_ice_properties=False,
        ),
        input_feature_names=FEATURES,
        add_static_features=False,
    )


def validate_dataset_config(config: DatasetConfig) -> None:
    """Check that `config` provides the features and pulsemap expected."""
    if list(config.features) != FEATURES:
        raise ValueError(f"Expected features in this order: {FEATURES}")
    pulsemaps = (
        [config.pulsemaps]
        if isinstance(config.pulsemaps, str)
        else list(config.pulsemaps)
    )
    if pulsemaps != ["SRTInIcePulses"]:
        raise ValueError("Expected the SRTInIcePulses pulsemap")


class IceMix(Model):
    """Shared CLS with vertex-only readout normalization."""

    def __init__(
        self,
        dropout: float = 0.1,
        attn_drop: float = 0.05,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.2,
    ) -> None:
        """Construct `IceMix`.

        Args:
            dropout: Dropout probability in the MLP of each block.
            attn_drop: Dropout probability on the attention weights.
            proj_drop: Dropout probability on the attention output
                projection of the relative-position blocks.
            drop_path_rate: Largest drop-path probability, reached by the
                last block of each stack.
        """
        super().__init__()
        for rate in (dropout, attn_drop, proj_drop, drop_path_rate):
            if not 0 <= rate < 1:
                raise ValueError("Dropout rates must lie in [0, 1)")
        self.fourier_ext = FourierEncoder(
            schema={0: 4096, 1: 4096, 2: 4096, 3: 4096, 4: 1024, 5: 1024},
            scaled=True,
        )
        width = self.fourier_ext.output_dim
        self.fourier_mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, 384),
        )
        self.rel_pos = SpacetimeEncoderEPJC(seq_length=32)
        self.sandwich = nn.ModuleList(
            [
                Block_rel(
                    input_dim=384,
                    num_heads=12,
                    mlp_ratio=4,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_path_rate * (i / 3),
                    init_values=1.0,
                )
                for i in range(4)
            ]
        )
        self.cls_token = nn.Linear(384, 1, bias=False)
        self.cls_norm = nn.LayerNorm(384)
        self.blocks = nn.ModuleList(
            [
                Block(
                    input_dim=384,
                    num_heads=12,
                    mlp_ratio=4,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    drop_path=drop_path_rate * (i / 11),
                    init_values=1.0,
                )
                for i in range(12)
            ]
        )

    @property
    def nb_outputs(self) -> int:
        """Return the width of each output."""
        return 384

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        """Return the normalised and the raw CLS readout."""
        x0, pulse_mask, lengths = array_to_sequence(
            data.x, data.batch, padding_value=0
        )
        x = self.fourier_mlp(self.fourier_ext(x0, lengths))
        rel_pos = self.rel_pos(x0)
        attn_mask = torch.zeros(pulse_mask.shape, device=pulse_mask.device)
        attn_mask[~pulse_mask] = -torch.inf
        for block in self.sandwich:
            x = block(x, attn_mask, rel_pos)
        batch_size = pulse_mask.shape[0]
        sequence_mask = torch.cat(
            [
                torch.ones(
                    batch_size,
                    1,
                    dtype=pulse_mask.dtype,
                    device=pulse_mask.device,
                ),
                pulse_mask,
            ],
            1,
        )
        attn_mask = torch.zeros(
            sequence_mask.shape, device=sequence_mask.device
        )
        attn_mask[~sequence_mask] = -torch.inf
        cls = self.cls_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([cls, x], 1)
        for block in self.blocks:
            x = block(x, None, attn_mask)
        shared = x[:, 0]
        # Vertex needs a stable readout scale; kappa retains the raw CLS scale.
        return self.cls_norm(shared), shared


class IceMixReconstruction(StandardModel):
    """Joint reconstruction with tunable training hyperparameters."""

    def __init__(
        self,
        vertex_weight: float = 0.026,
        direction_weight: float = 1.0,
        lr: float = 6.25e-5,
        weight_decay: float = 0.01,
        adam_eps: float = 1e-5,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 3,
        dropout: float = 0.1,
        attn_drop: float = 0.05,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.2,
    ) -> None:
        """Construct `IceMixReconstruction`.

        Args:
            vertex_weight: Weight of the vertex loss, in metres.
            direction_weight: Weight of the direction loss.
            lr: Learning rate of `AdamW`.
            weight_decay: Weight decay of `AdamW`.
            adam_eps: Epsilon of `AdamW`.
            scheduler_factor: Factor of `ReduceLROnPlateau`.
            scheduler_patience: Patience of `ReduceLROnPlateau`.
            dropout: See `IceMix`.
            attn_drop: See `IceMix`.
            proj_drop: See `IceMix`.
            drop_path_rate: See `IceMix`.
        """
        for value in (
            vertex_weight,
            direction_weight,
            lr,
            weight_decay,
            adam_eps,
            scheduler_factor,
        ):
            if not math.isfinite(value):
                raise ValueError("Training settings must be finite")
        if lr <= 0 or adam_eps <= 0 or weight_decay < 0:
            raise ValueError(
                "Learning rate and epsilon must be positive; weight decay "
                "nonnegative"
            )
        if (
            not 0 < scheduler_factor < 1
            or not isinstance(scheduler_patience, int)
            or scheduler_patience < 0
        ):
            raise ValueError(
                "Scheduler factor must lie in (0, 1) and patience be a "
                "nonnegative integer"
            )
        if vertex_weight < 0 or direction_weight <= 0:
            raise ValueError(
                "Vertex weight must be nonnegative and direction weight "
                "positive"
            )
        backbone = IceMix(dropout, attn_drop, proj_drop, drop_path_rate)
        position_task = PositionReconstruction(
            hidden_size=384,
            target_labels=POSITION_LABELS,
            loss_function=PerEventRecordingLoss(
                EnsembleLoss(
                    [EuclideanDistanceLoss()], loss_factors=[vertex_weight]
                ),
                metric_name="position",
            ),
        )
        direction_loss = VonMisesFisher3DLoss()
        if direction_weight != 1.0:
            direction_loss = EnsembleLoss(
                [direction_loss], loss_factors=[direction_weight]
            )
        direction_task = DirectionReconstructionWithLogKappa(
            hidden_size=384,
            target_labels=["direction"],
            loss_function=PerEventRecordingLoss(
                direction_loss, metric_name="direction"
            ),
        )
        super().__init__(
            data_representation=build_data_representation(),
            backbone=backbone,
            tasks=[position_task, direction_task],
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": lr,
                "eps": adam_eps,
                "weight_decay": weight_decay,
            },
            scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_kwargs={
                "mode": "min",
                "patience": scheduler_patience,
                "factor": scheduler_factor,
            },
            scheduler_config={"monitor": "val_loss", "frequency": 1},
        )
