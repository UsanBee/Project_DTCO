"""Layer- and lane-level workload plans for the prefetch hierarchy.

The existing workload module remains the authority for D-2 attention/MLP
semantic work.  This module adds the coarser full-transformer-layer unit used
by the dedicated full-layer-ready prefetch engine and uniformly stripes each
TP-sharded layer across eight physical lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from stage1_model import MODELS, b_kv_footprint, b_weight_footprint

from .workload import build_decode_signature, build_prefill_signature


PREFETCH_LANE_COUNT = 8


@dataclass(frozen=True)
class LaneLayerPlan:
    """One lane's bytes for one full transformer layer."""

    lane_index: int
    layer_index: int
    weight_bytes: float
    historical_kv_bytes: float
    new_kv_bytes: float


@dataclass(frozen=True)
class LayerFootprint:
    """Per-GPU state and D-2 semantic work for one transformer layer."""

    layer_index: int
    weight_bytes: float
    historical_kv_bytes: float
    new_kv_bytes: float
    attention_compute_flops: float
    mlp_compute_flops: float
    attention_collective_bytes: float
    mlp_collective_bytes: float
    lanes: tuple[LaneLayerPlan, ...]


@dataclass(frozen=True)
class WeightResidencyDecision:
    """Typed capacity-policy result populated by the residency-policy layer."""

    feasible: bool
    requested_policy: str
    effective_policy: str | None
    reason: str | None = None
    mandatory_local_bytes_per_lane: float = 0.0
    persistent_weight_bytes_per_lane: float = 0.0
    streaming_weight_buffer_bytes_per_lane: float = 0.0
    side_kv_bytes_per_lane: float = 0.0
    first_infeasible_token: int | None = None


def split_residual(total: float, count: int) -> tuple[float, ...]:
    """Split a nonnegative total and conserve it under :func:`math.fsum`.

    The first ``count - 1`` values are the ordinary equal share.  The final
    value carries the representational residual, avoiding byte loss when a
    total is decomposed repeatedly across layers and lanes.
    """
    if not math.isfinite(total) or total < 0.0:
        raise ValueError("split total must be nonnegative and finite")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("split count must be a positive integer")
    if count == 1:
        return (total,)
    share = total / count
    parts = [share] * (count - 1)
    parts.append(total - math.fsum(parts))
    return tuple(parts)


def _lane_plans(
    layer_index: int,
    weight_bytes: float,
    historical_kv_bytes: float,
    new_kv_bytes: float,
    lane_count: int,
) -> tuple[LaneLayerPlan, ...]:
    weights = split_residual(weight_bytes, lane_count)
    historical = split_residual(historical_kv_bytes, lane_count)
    generated = split_residual(new_kv_bytes, lane_count)
    return tuple(
        LaneLayerPlan(
            lane_index=lane,
            layer_index=layer_index,
            weight_bytes=weights[lane],
            historical_kv_bytes=historical[lane],
            new_kv_bytes=generated[lane],
        )
        for lane in range(lane_count)
    )


def build_layer_plans(
    model_name: str,
    tp: int,
    seq_len: int,
    batch: int = 1,
    *,
    phase: str,
    decode_tokens: int = 1,
    token_index: int = 0,
    lane_count: int = PREFETCH_LANE_COUNT,
) -> tuple[LayerFootprint, ...]:
    """Build one prefill or decode step as full-layer-ready lane plans.

    ``token_index`` is zero-based within a decode run.  Historical KV for that
    step contains the configured prefix plus every previously completed decode
    token.  Fresh prefill has no historical-KV prefetch; its prompt KV appears
    only in ``new_kv_bytes`` and is committed after attention.
    """
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name}")
    if tp <= 0 or seq_len <= 0 or batch <= 0:
        raise ValueError("tp, seq_len, and batch must be positive")
    if lane_count != PREFETCH_LANE_COUNT:
        raise ValueError(
            f"layer-prefetch workload requires exactly {PREFETCH_LANE_COUNT} lanes"
        )
    if phase not in {"prefill", "decode"}:
        raise ValueError("phase must be prefill or decode")
    if decode_tokens <= 0:
        raise ValueError("decode_tokens must be positive")
    if token_index < 0:
        raise ValueError("token_index must be nonnegative")
    if phase == "decode" and token_index >= decode_tokens:
        raise ValueError("token_index must be smaller than decode_tokens")
    if phase == "prefill" and token_index != 0:
        raise ValueError("prefill does not accept a nonzero token_index")

    model = MODELS[model_name]
    n_layers = model["n_layers"]
    weight_total = b_weight_footprint(model, tp)

    if phase == "prefill":
        historical_kv_total = 0.0
        new_kv_total = b_kv_footprint(model, tp, seq_len, batch)
        signature = build_prefill_signature(model_name, tp, seq_len, batch)
    else:
        current_seq_len = seq_len + token_index
        historical_kv_total = b_kv_footprint(
            model, tp, current_seq_len, batch
        )
        new_kv_total = (
            b_kv_footprint(model, tp, current_seq_len + 1, batch)
            - historical_kv_total
        )
        signature = build_decode_signature(
            model_name, tp, current_seq_len, batch
        )

    if signature.semantic_stages is None:
        raise RuntimeError("LLM workload signature lacks D-2 semantic stages")

    weights = split_residual(weight_total, n_layers)
    historical = split_residual(historical_kv_total, n_layers)
    generated = split_residual(new_kv_total, n_layers)

    plans = []
    for layer_index in range(n_layers):
        attention = signature.semantic_stages[2 * layer_index]
        mlp = signature.semantic_stages[2 * layer_index + 1]
        plans.append(
            LayerFootprint(
                layer_index=layer_index,
                weight_bytes=weights[layer_index],
                historical_kv_bytes=historical[layer_index],
                new_kv_bytes=generated[layer_index],
                attention_compute_flops=attention.compute_flops,
                mlp_compute_flops=mlp.compute_flops,
                attention_collective_bytes=attention.collective_bytes,
                mlp_collective_bytes=mlp.collective_bytes,
                lanes=_lane_plans(
                    layer_index,
                    weights[layer_index],
                    historical[layer_index],
                    generated[layer_index],
                    lane_count,
                ),
            )
        )
    return tuple(plans)
