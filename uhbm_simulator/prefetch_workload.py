"""Layer- and lane-level workload plans for the prefetch hierarchy.

The existing workload module remains the authority for D-2 attention/MLP
semantic work.  This module adds the coarser full-transformer-layer unit used
by the dedicated full-layer-ready prefetch engine and uniformly stripes each
TP-sharded layer across eight physical lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from stage1_model import MODELS, b_kv_footprint, b_weight_footprint

from .workload import build_decode_signature, build_prefill_signature
from .prefetch_config import (
    ArchitectureConfig,
    PrefetchPolicyConfig,
    PrefetchWorkloadConfig,
)


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


class InfeasibilityReason(Enum):
    """Typed reasons for rejecting one explicit hierarchy point."""

    LOCAL_MANDATORY_STATE = "local_mandatory_state"
    LOCAL_WEIGHT_BUFFERS = "local_weight_buffers"
    SIDE_KV_CAPACITY = "side_kv_capacity"


@dataclass(frozen=True)
class WeightResidencyDecision:
    """Typed capacity-policy result populated by the residency-policy layer."""

    feasible: bool
    requested_policy: str
    effective_policy: str | None
    reason: InfeasibilityReason | None = None
    mandatory_local_bytes_per_lane: float = 0.0
    persistent_weight_bytes_per_lane: float = 0.0
    streaming_weight_buffer_bytes_per_lane: float = 0.0
    side_kv_bytes_per_lane: float = 0.0
    first_infeasible_token: int | None = None


def _single(values: tuple, name: str):
    if len(values) != 1:
        raise ValueError(
            f"{name} must be an expanded sweep point with a single value"
        )
    return values[0]


@dataclass(frozen=True)
class _TokenCapacityState:
    token_number: int
    mandatory_local_bytes_per_lane: float
    side_kv_bytes_per_lane: float
    max_weight_layer_bytes_per_lane: float
    total_weight_bytes_per_lane: float


def _capacity_state(
    architecture: ArchitectureConfig,
    workload: PrefetchWorkloadConfig,
    *,
    phase: str,
    token_index: int,
    decode_tokens: int,
) -> _TokenCapacityState:
    plans = build_layer_plans(
        workload.model,
        workload.tp,
        workload.seq_len,
        _single(workload.batch_values, "workload.batch"),
        phase=phase,
        decode_tokens=decode_tokens,
        token_index=token_index,
        lane_count=architecture.lane_count,
    )
    lane = 0
    total_weight = math.fsum(plan.lanes[lane].weight_bytes for plan in plans)
    max_weight_layer = max(plan.lanes[lane].weight_bytes for plan in plans)
    historical_total = math.fsum(
        plan.lanes[lane].historical_kv_bytes for plan in plans
    )
    new_total = math.fsum(plan.lanes[lane].new_kv_bytes for plan in plans)
    max_historical_layer = max(
        plan.lanes[lane].historical_kv_bytes for plan in plans
    )
    max_new_layer = max(plan.lanes[lane].new_kv_bytes for plan in plans)

    rounds = 0 if workload.tp <= 1 else 2 * (workload.tp - 1)
    if rounds == 0:
        collective_scratch = 0.0
    else:
        collective_scratch = max(
            max(
                plan.attention_collective_bytes,
                plan.mlp_collective_bytes,
            )
            / rounds
            / architecture.lane_count
            for plan in plans
        )

    side_kv = historical_total + new_total
    if architecture.kind == "b2_uhbm":
        # Two full layer buffers preserve current/next readiness.  The newly
        # generated layer KV remains live until its writeback commits, and the
        # per-round collective chunk is the reusable materialized scratch.
        mandatory_local = (
            2.0 * max_historical_layer
            + max_new_layer
            + collective_scratch
        )
    elif architecture.kind == "b0_2_5d":
        # Conventional HBM is the KV backing tier itself, so all resident KV
        # through this step is mandatory local state.
        mandatory_local = side_kv + collective_scratch
        side_kv = 0.0
    else:
        raise ValueError(f"unknown prefetch architecture kind: {architecture.kind}")

    return _TokenCapacityState(
        token_number=(0 if phase == "prefill" else token_index + 1),
        mandatory_local_bytes_per_lane=mandatory_local,
        side_kv_bytes_per_lane=side_kv,
        max_weight_layer_bytes_per_lane=max_weight_layer,
        total_weight_bytes_per_lane=total_weight,
    )


def _capacity_timeline(
    architecture: ArchitectureConfig,
    workload: PrefetchWorkloadConfig,
) -> tuple[_TokenCapacityState, ...]:
    run_mode = _single(workload.run_mode_values, "workload.run_mode")
    start_state = _single(workload.start_state_values, "workload.start_state")
    decode_tokens = _single(workload.decode_tokens_values, "workload.decode_tokens")
    if decode_tokens <= 0:
        raise ValueError("workload.decode_tokens must be positive")
    decode_steps = decode_tokens + (
        1
        if start_state == "steady_state"
        and run_mode in {"decode_only", "prefill_decode"}
        else 0
    )

    states = []
    if run_mode in {"prefill_only", "prefill_decode"}:
        states.append(
            _capacity_state(
                architecture,
                workload,
                phase="prefill",
                token_index=0,
                decode_tokens=decode_tokens,
            )
        )
    if run_mode in {"decode_only", "prefill_decode"}:
        states.extend(
            _capacity_state(
                architecture,
                workload,
                phase="decode",
                token_index=token_index,
                decode_tokens=decode_steps,
            )
            for token_index in range(decode_steps)
        )
    if not states:
        raise ValueError(f"unsupported workload.run_mode: {run_mode}")
    return tuple(states)


def decide_weight_residency(
    architecture: ArchitectureConfig,
    workload: PrefetchWorkloadConfig,
    policy: PrefetchPolicyConfig,
) -> WeightResidencyDecision:
    """Reserve mandatory state, then choose whole-model persistence/streaming.

    Inputs must already be expanded to one sweep point.  Capacity-memory
    backing is not considered local execution capacity, and no legacy
    ``r_remote`` or ``capacity_overflow`` semantics are reused here.
    """
    requested_policy = _single(
        policy.weight_residency_policy_values,
        "prefetch.weight_residency_policy",
    )
    _single(policy.weight_chunk_bytes_values, "prefetch.weight_chunk_bytes")
    _single(policy.kv_chunk_bytes_values, "prefetch.kv_chunk_bytes")
    _single(
        policy.kv_writeback_chunk_bytes_values,
        "prefetch.kv_writeback_chunk_bytes",
    )
    _single(
        policy.max_inflight_chunks_per_lane_values,
        "prefetch.max_inflight_chunks_per_lane",
    )
    if requested_policy not in {"persistent_if_fit", "streaming"}:
        raise ValueError(f"unknown weight residency policy: {requested_policy}")

    states = _capacity_timeline(architecture, workload)
    peak_mandatory = max(
        state.mandatory_local_bytes_per_lane for state in states
    )
    final_side_kv = max(state.side_kv_bytes_per_lane for state in states)
    total_weight = states[0].total_weight_bytes_per_lane
    streaming_buffers = 2.0 * max(
        state.max_weight_layer_bytes_per_lane for state in states
    )

    if architecture.kind == "b2_uhbm":
        if architecture.side_hbm is None:
            raise ValueError("b2_uhbm requires paired side_hbm capacity")
        for state in states:
            if state.side_kv_bytes_per_lane > architecture.side_hbm.capacity_bytes:
                return WeightResidencyDecision(
                    feasible=False,
                    requested_policy=requested_policy,
                    effective_policy=None,
                    reason=InfeasibilityReason.SIDE_KV_CAPACITY,
                    mandatory_local_bytes_per_lane=peak_mandatory,
                    streaming_weight_buffer_bytes_per_lane=streaming_buffers,
                    side_kv_bytes_per_lane=final_side_kv,
                    first_infeasible_token=state.token_number,
                )

    if requested_policy == "persistent_if_fit" and (
        peak_mandatory + total_weight
        <= architecture.local_memory.capacity_bytes
    ):
        return WeightResidencyDecision(
            feasible=True,
            requested_policy=requested_policy,
            effective_policy="persistent",
            mandatory_local_bytes_per_lane=peak_mandatory,
            persistent_weight_bytes_per_lane=total_weight,
            side_kv_bytes_per_lane=final_side_kv,
        )

    effective_policy = "streaming"
    for state in states:
        required = (
            state.mandatory_local_bytes_per_lane + streaming_buffers
        )
        if required > architecture.local_memory.capacity_bytes:
            reason = (
                InfeasibilityReason.LOCAL_MANDATORY_STATE
                if state.mandatory_local_bytes_per_lane
                > architecture.local_memory.capacity_bytes
                else InfeasibilityReason.LOCAL_WEIGHT_BUFFERS
            )
            return WeightResidencyDecision(
                feasible=False,
                requested_policy=requested_policy,
                effective_policy=effective_policy,
                reason=reason,
                mandatory_local_bytes_per_lane=peak_mandatory,
                streaming_weight_buffer_bytes_per_lane=streaming_buffers,
                side_kv_bytes_per_lane=final_side_kv,
                first_infeasible_token=state.token_number,
            )

    return WeightResidencyDecision(
        feasible=True,
        requested_policy=requested_policy,
        effective_policy=effective_policy,
        mandatory_local_bytes_per_lane=peak_mandatory,
        streaming_weight_buffer_bytes_per_lane=streaming_buffers,
        side_kv_bytes_per_lane=final_side_kv,
    )


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
