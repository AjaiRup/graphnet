"""Transformer for joint direction and vertex reconstruction."""

from typing import Optional, Tuple
import math

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from graphnet.models import Model, StandardModel
from graphnet.models.components.embedding import (
    SinusoidalPosEmb,
    SpacetimeEncoder,
)
from graphnet.models.components.layers import Block, Block_rel
from graphnet.models.data_representation import GraphDefinition
from graphnet.models.data_representation.graphs.nodes import IceMixNodes
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.task.task import IdentityTask
from graphnet.models.task.reconstruction import (
    DirectionReconstructionWithKappaDecoupled,
)
from graphnet.models.utils import array_to_sequence
from graphnet.training.loss_functions import (
    EnsembleLoss,
    EuclideanDistanceLoss,
    PerEventRecordingLoss,
    VonMisesFisher3DLoss,
)
from graphnet.utilities.config import ModelConfig

FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "rde"]
POSITION_LABELS = ["position_x", "position_y", "position_z"]


def scale_position(x: torch.Tensor) -> torch.Tensor:
    return x / 100.0


def unscale_position(x: torch.Tensor) -> torch.Tensor:
    return x * 100.0


def build_data_representation() -> GraphDefinition:
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


def validate_dataset_config(config) -> None:
    if list(config.features) != FEATURES:
        raise ValueError(f"Expected features in this order: {FEATURES}")
    pulsemaps = (
        [config.pulsemaps]
        if isinstance(config.pulsemaps, str)
        else list(config.pulsemaps)
    )
    if pulsemaps != ["SRTInIcePulses"]:
        raise ValueError("Expected the SRTInIcePulses pulsemap")


class _RDEEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=128, scaled=True)
        self.sin_emb2 = SinusoidalPosEmb(dim=64, scaled=True)
        self.mlp = nn.Sequential(
            nn.Linear(832, 832),
            nn.LayerNorm(832),
            nn.GELU(),
            nn.Linear(832, 384),
        )

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        length = torch.log10(lengths.to(dtype=x.dtype))
        embeddings = [
            self.sin_emb(4096.0 * x[:, :, :3]).flatten(-2),
            self.sin_emb(1024.0 * x[:, :, 4]),
            self.sin_emb(4096.0 * x[:, :, 3]),
            self.sin_emb(1024.0 * x[:, :, 5]),
            self.sin_emb2(length).unsqueeze(1).expand(-1, x.shape[1], -1),
        ]
        return self.mlp(torch.cat(embeddings, -1))


class IceMix(Model):
    """Shared CLS with vertex-only readout normalization."""

    def __init__(
        self,
        dropout: float = 0.1,
        attn_drop: float = 0.05,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        for rate in (dropout, attn_drop, proj_drop, drop_path_rate):
            if not 0 <= rate < 1:
                raise ValueError("Dropout rates must lie in [0, 1)")
        self.fourier_ext = _RDEEncoder()
        self.rel_pos = SpacetimeEncoder(seq_length=32)
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
        return 384

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x0, pulse_mask, lengths = array_to_sequence(
            data.x, data.batch, padding_value=0
        )
        x = self.fourier_ext(x0, lengths)
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
        vertex_weight: float = 2.6,
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
                "Learning rate and epsilon must be positive; weight decay nonnegative"
            )
        if (
            not 0 < scheduler_factor < 1
            or not isinstance(scheduler_patience, int)
            or scheduler_patience < 0
        ):
            raise ValueError(
                "Scheduler factor must lie in (0, 1) and patience be a nonnegative integer"
            )
        if vertex_weight < 0 or direction_weight <= 0:
            raise ValueError(
                "Vertex weight must be nonnegative and direction weight positive"
            )
        backbone = IceMix(dropout, attn_drop, proj_drop, drop_path_rate)
        position_task = IdentityTask(
            nb_outputs=3,
            target_labels=POSITION_LABELS,
            prediction_labels=[f"position_{axis}_pred" for axis in "xyz"],
            hidden_size=384,
            loss_function=PerEventRecordingLoss(
                EnsembleLoss(
                    [EuclideanDistanceLoss()], loss_factors=[vertex_weight]
                ),
                metric_name="position",
            ),
            transform_target=scale_position,
            transform_inference=unscale_position,
        )
        direction_loss = VonMisesFisher3DLoss(
            kappa_switch=700.0, approximation="tight"
        )
        if direction_weight != 1.0:
            direction_loss = EnsembleLoss(
                [direction_loss], loss_factors=[direction_weight]
            )
        direction_task = DirectionReconstructionWithKappaDecoupled(
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


def load_icemix_model(
    path: Optional[str] = None, **overrides
) -> IceMixReconstruction:
    settings = {}
    if path is not None:
        config = ModelConfig.load(path)
        if config.class_name != "IceMixReconstruction":
            raise ValueError("Expected an IceMixReconstruction config")
        settings.update(config.arguments)
    settings.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    return IceMixReconstruction(**settings)
