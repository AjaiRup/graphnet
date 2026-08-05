"""Implementation of IceMix.

This method was a solution submitted to the IceCube - Neutrinos in Deep Ice
Kaggle competition.

Solution by DrHB: https://github.com/DrHB/icecube-2nd-place
"""

import torch
import torch.nn as nn
from typing import Set, Dict, Any, Optional

from graphnet.models.components.layers import (
    Block_rel,
    Block,
)
from graphnet.models.components.embedding import (
    FourierEncoder,
    SpacetimeEncoder,
    MahalanobisEncoder,
)
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import array_to_sequence

from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch import Tensor


def _drop_path_at(index: int, depth: int, drop_path_rate: float) -> float:
    """Return the drop-path rate of block `index` of a `depth`-block stack.

    Scales linearly from 0.0 at the first block to `drop_path_rate` at the
    last.
    """
    if depth <= 1:
        return 0.0
    return drop_path_rate * (index / (depth - 1))


class DeepIce(GNN):
    """DeepIce model."""

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_ratio: int = 4,
        seq_length: int = 192,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        n_rel: int = 1,
        scaled_emb: bool = False,
        include_dynedge: bool = False,
        dynedge_args: Optional[Dict[str, Any]] = None,
        n_features: int = 6,
        init_values: Optional[float] = 1.0,
        init_values_rel: Optional[float] = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        token_drop: float = 0.0,
        drop_chance: float = 1.0,
        eval_token_drop: bool = False,
        relative_encoder: str = "spacetime",
        pos_time_multiplier: float = 4096.0,
        charge_multiplier: float = 1024.0,
        distance_scale: float = 3e4 / 500 * 3e-1,
        distance_clip_min: float = -4.0,
        distance_clip_max: float = 4.0,
        distance_multiplier: float = 1024.0,
        n_freq: float = 10000.0,
    ):
        """Construct `DeepIce`.

        Args:
            hidden_dim: The latent feature dimension.
            mlp_ratio: Mlp expansion ratio of FourierEncoder and Transformer.
            seq_length: The base feature dimension.
            depth: The depth of the transformer.
            head_size: The size of the attention heads.
            depth_rel: The depth of the relative transformer.
            n_rel: The number of relative transformer layers to use.
            scaled_emb: Whether to scale the sinusoidal positional embeddings.
            include_dynedge: If True, pulse-level predictions from `DynEdge`
                will be added as features to the model.
            dynedge_args: Initialization arguments for DynEdge. If not
                provided, DynEdge will be initialized with the original Kaggle
                Competition settings. If `include_dynedge` is False, this
                argument have no impact.
            n_features: The number of features in the input data.
            init_values: Initial value of the LayerScale parameters of each
                `Block`. If None, LayerScale is disabled.
            init_values_rel: Initial value of the LayerScale parameters of each
                relative-attention `Block_rel`. If None, LayerScale is disabled
                there - which is the original Kaggle solution's behaviour.
            dropout: Dropout probability of the `Mlp` layer of each block.
            attn_drop: Dropout probability of the attention weights.
            proj_drop: Dropout probability of the attention output projection.
            drop_path_rate: Maximum drop-path (stochastic depth) rate. Scaled
                linearly from 0.0 to `drop_path_rate` across the depth of the
                model.
            token_drop: Probability with which each individual pulse is dropped
                from an affected event during training. Regularizes the model
                against missing pulses.
            drop_chance: Probability that a given event is affected by token
                dropping at all. Only has an effect if `token_drop > 0`.
            eval_token_drop: If True, token dropping is also applied outside
                training. Intended for robustness evaluation, where the model
                is deliberately probed at a range of `token_drop` values.
            relative_encoder: Which relative positional encoder to use, either
                `"spacetime"` (Minkowski four-distance, as in the 2nd place
                solution) or `"mahalanobis"` (positive-definite metric).
            pos_time_multiplier: Scale applied to position and time features
                before sinusoidal encoding.
            charge_multiplier: Scale applied to the charge feature before
                sinusoidal encoding.
            distance_scale: Conversion factor applied to the time difference in
                the relative positional encoder.
            distance_clip_min: Lower clip applied to the relative distance.
            distance_clip_max: Upper clip applied to the relative distance.
            distance_multiplier: Scale applied to the clipped relative distance
                before sinusoidal encoding.
            n_freq: Number of frequencies of the sinusoidal embeddings.
        """
        super().__init__(seq_length, hidden_dim)
        if relative_encoder not in ("spacetime", "mahalanobis"):
            raise ValueError(
                "`relative_encoder` must be either 'spacetime' or "
                f"'mahalanobis'. Got '{relative_encoder}'."
            )
        if not 0.0 <= token_drop <= 1.0:
            raise ValueError(
                f"`token_drop` must be in [0, 1]. Got {token_drop}."
            )
        if not 0.0 <= drop_chance <= 1.0:
            raise ValueError(
                f"`drop_chance` must be in [0, 1]. Got {drop_chance}."
            )

        self.token_drop = token_drop
        self.drop_chance = drop_chance
        self.eval_token_drop = eval_token_drop

        fourier_out_dim = hidden_dim // 2 if include_dynedge else hidden_dim
        self.fourier_ext = FourierEncoder(
            seq_length=seq_length,
            mlp_dim=None,
            output_dim=fourier_out_dim,
            scaled=scaled_emb,
            n_features=n_features,
            pos_time_multiplier=pos_time_multiplier,
            charge_multiplier=charge_multiplier,
            n_freq=n_freq,
        )
        encoder_class = (
            MahalanobisEncoder
            if relative_encoder == "mahalanobis"
            else SpacetimeEncoder
        )
        self.rel_pos = encoder_class(
            seq_length=head_size,
            distance_scale=distance_scale,
            distance_clip_min=distance_clip_min,
            distance_clip_max=distance_clip_max,
            distance_multiplier=distance_multiplier,
            n_freq=n_freq,
        )
        self.sandwich = nn.ModuleList(
            [
                Block_rel(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=_drop_path_at(i, depth_rel, drop_path_rate),
                    init_values=init_values_rel,
                )
                for i in range(depth_rel)
            ]
        )
        self.cls_token = nn.Linear(hidden_dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    drop_path=_drop_path_at(i, depth, drop_path_rate),
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.n_rel = n_rel

        if include_dynedge and dynedge_args is None:
            self.warning_once("Running with default DynEdge settings")
            self.dyn_edge = DynEdge(
                nb_inputs=9,
                nb_neighbours=9,
                post_processing_layer_sizes=[336, hidden_dim // 2],
                dynedge_layer_sizes=[
                    (128, 256),
                    (336, 256),
                    (336, 256),
                    (336, 256),
                ],
                global_pooling_schemes=None,
                activation_layer="gelu",
                add_norm_layer=True,
                skip_readout=True,
            )
        elif include_dynedge and not (dynedge_args is None):
            self.dyn_edge = DynEdge(**dynedge_args)

        self.include_dynedge = include_dynedge

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    def _drop_tokens(self, data: Data) -> Data:
        """Randomly drop pulses from a subset of the events in `data`.

        Each event is selected for dropping with probability `drop_chance`;
        within a selected event, each pulse is dropped with probability
        `token_drop`. At least one pulse is always retained per event - an
        emptied event would silently disappear from `Data.batch`, shrinking the
        effective batch and misaligning predictions against their targets.
        """
        n_events = int(data.batch.max().item()) + 1
        device = data.x.device

        event_is_affected = (
            torch.rand(n_events, device=device) < self.drop_chance
        )
        pulse_is_dropped = (
            torch.rand(data.x.shape[0], device=device) < self.token_drop
        )
        keep_mask = ~(event_is_affected[data.batch] & pulse_is_dropped)

        # `Batch.from_data_list` emits a sorted `batch`, so the first row of
        # each event sits at the exclusive cumulative sum of the event sizes.
        counts = torch.bincount(data.batch, minlength=n_events)
        first_row = torch.cumsum(counts, dim=0) - counts
        emptied = (
            torch.bincount(data.batch[keep_mask], minlength=n_events) == 0
        ) & (counts > 0)
        if bool(emptied.any()):
            keep_mask[first_row[emptied]] = True

        data.x = data.x[keep_mask]
        data.batch = data.batch[keep_mask]
        if hasattr(data, "n_pulses"):
            data.n_pulses = torch.bincount(data.batch, minlength=n_events).to(
                data.n_pulses.dtype
            )
        return data

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        if (
            self.token_drop > 0.0
            and self.drop_chance > 0.0
            and (self.training or self.eval_token_drop)
        ):
            data = self._drop_tokens(data)

        x0, mask, seq_length = array_to_sequence(
            data.x, data.batch, padding_value=0
        )
        x = self.fourier_ext(x0, seq_length)
        rel_pos_bias = self.rel_pos(x0)
        batch_size = mask.shape[0]
        if self.include_dynedge:
            graph = self.dyn_edge(data)
            graph, _ = to_dense_batch(graph, data.batch)
            x = torch.cat([x, graph], 2)

        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        mask = torch.cat(
            [
                torch.ones(
                    batch_size, 1, dtype=mask.dtype, device=mask.device
                ),
                mask,
            ],
            1,
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            x = blk(x, None, attn_mask)

        return x[:, 0]
