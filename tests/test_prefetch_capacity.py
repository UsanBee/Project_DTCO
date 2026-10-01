from dataclasses import replace
import math

import pytest

from uhbm_simulator.prefetch_config import (
    ArchitectureConfig,
    MemoryDeviceConfig,
    PrefetchPolicyConfig,
    PrefetchWorkloadConfig,
)
from uhbm_simulator.prefetch_workload import build_layer_plans


def _memory(capacity_bytes):
    return MemoryDeviceConfig(
        capacity_bytes=capacity_bytes,
        bw_bps=1.0,
        latency_s=0.0,
        energy_pj_per_bit=1.0,
    )


def _architecture(kind, local_capacity, side_capacity=None):
    return ArchitectureConfig(
        name=kind,
        kind=kind,
        lane_count=8,
        local_memory=_memory(local_capacity),
        side_hbm=(None if side_capacity is None else _memory(side_capacity)),
        side_link=None,
    )


def _workload(*, decode_tokens=4):
    return PrefetchWorkloadConfig(
        run_mode_values=("decode_only",),
        start_state_values=("cold_start",),
        model="Llama-70B",
        tp=4,
        pp=1,
        seq_len=4096,
        batch_values=(1,),
        decode_tokens_values=(decode_tokens,),
    )


def _policy(name):
    return PrefetchPolicyConfig(
        weight_residency_policy_values=(name,),
        weight_chunk_bytes_values=(1024,),
        kv_chunk_bytes_values=(512,),
        kv_writeback_chunk_bytes_values=(256,),
        max_inflight_chunks_per_lane_values=(1,),
    )


def _per_lane_requirements(decode_tokens=4):
    plans = build_layer_plans(
        "Llama-70B",
        4,
        4096,
        1,
        phase="decode",
        decode_tokens=decode_tokens,
        token_index=decode_tokens - 1,
    )
    weight = math.fsum(lane.weight_bytes for plan in plans for lane in plan.lanes if lane.lane_index == 0)
    max_weight_layer = max(plan.lanes[0].weight_bytes for plan in plans)
    max_historical_layer = max(
        plan.lanes[0].historical_kv_bytes for plan in plans
    )
    max_new_layer = max(plan.lanes[0].new_kv_bytes for plan in plans)
    rounds = 2 * (4 - 1)
    scratch = max(
        max(plan.attention_collective_bytes, plan.mlp_collective_bytes)
        / rounds
        / 8.0
        for plan in plans
    )
    final_kv = math.fsum(plan.lanes[0].historical_kv_bytes for plan in plans) + math.fsum(
        plan.lanes[0].new_kv_bytes for plan in plans
    )
    b2_mandatory = 2.0 * max_historical_layer + max_new_layer + scratch
    b0_mandatory = final_kv + scratch
    return {
        "weight": weight,
        "max_weight_layer": max_weight_layer,
        "scratch": scratch,
        "final_kv": final_kv,
        "b2_mandatory": b2_mandatory,
        "b0_mandatory": b0_mandatory,
    }


def test_b2_reserves_current_next_kv_and_collective_scratch_before_weights():
    """Checking weight fit before mandatory uHBM state would fail this test."""
    from uhbm_simulator.prefetch_workload import decide_weight_residency

    required = _per_lane_requirements()
    capacity = required["b2_mandatory"] + required["weight"]
    result = decide_weight_residency(
        _architecture("b2_uhbm", capacity, required["final_kv"]),
        _workload(),
        _policy("persistent_if_fit"),
    )

    assert result.feasible
    assert result.effective_policy == "persistent"
    assert result.mandatory_local_bytes_per_lane == pytest.approx(
        required["b2_mandatory"]
    )
    assert result.persistent_weight_bytes_per_lane == required["weight"]


def test_b0_reserves_full_historical_kv_before_weight_persistence():
    """Treating B0 KV as a transient layer buffer would incorrectly persist."""
    from uhbm_simulator.prefetch_workload import decide_weight_residency

    required = _per_lane_requirements()
    capacity = required["b0_mandatory"] + required["weight"]
    result = decide_weight_residency(
        _architecture("b0_2_5d", capacity),
        _workload(),
        _policy("persistent_if_fit"),
    )

    assert result.feasible
    assert result.effective_policy == "persistent"
    assert result.mandatory_local_bytes_per_lane == pytest.approx(
        required["b0_mandatory"]
    )


def test_one_byte_short_of_persistent_fit_falls_back_atomically_to_streaming():
    """Partial weight pinning instead of all-or-nothing fallback must fail."""
    from uhbm_simulator.prefetch_workload import decide_weight_residency

    required = _per_lane_requirements()
    capacity = required["b2_mandatory"] + required["weight"] - 1.0
    result = decide_weight_residency(
        _architecture("b2_uhbm", capacity, required["final_kv"]),
        _workload(),
        _policy("persistent_if_fit"),
    )

    assert result.feasible
    assert result.effective_policy == "streaming"
    assert result.persistent_weight_bytes_per_lane == 0.0
    assert result.streaming_weight_buffer_bytes_per_lane == pytest.approx(
        2.0 * required["max_weight_layer"]
    )


def test_streaming_is_infeasible_when_double_weight_buffer_does_not_fit():
    """Allowing a single layer buffer would violate next-layer prefetching."""
    from uhbm_simulator.prefetch_workload import (
        InfeasibilityReason,
        decide_weight_residency,
    )

    required = _per_lane_requirements()
    capacity = (
        required["b2_mandatory"]
        + 2.0 * required["max_weight_layer"]
        - 1.0
    )
    result = decide_weight_residency(
        _architecture("b2_uhbm", capacity, required["final_kv"]),
        _workload(),
        _policy("streaming"),
    )

    assert not result.feasible
    assert result.reason is InfeasibilityReason.LOCAL_WEIGHT_BUFFERS


def test_b2_side_hbm_overflow_is_typed_and_not_spilled_to_capacity_memory():
    """KV eviction into weight backing memory is outside the v1 contract."""
    from uhbm_simulator.prefetch_workload import (
        InfeasibilityReason,
        decide_weight_residency,
    )

    required = _per_lane_requirements()
    result = decide_weight_residency(
        _architecture("b2_uhbm", 1e12, required["final_kv"] - 1.0),
        _workload(),
        _policy("streaming"),
    )

    assert not result.feasible
    assert result.reason is InfeasibilityReason.SIDE_KV_CAPACITY
    assert result.side_kv_bytes_per_lane == pytest.approx(required["final_kv"])


def test_capacity_growth_reports_first_infeasible_decode_token():
    """Reporting only final overflow would hide the causal failure boundary."""
    from uhbm_simulator.prefetch_workload import (
        InfeasibilityReason,
        decide_weight_residency,
    )

    workload = _workload(decode_tokens=3)
    token_zero = build_layer_plans(
        "Llama-70B", 4, 4096, 1, phase="decode", decode_tokens=3, token_index=0
    )
    initial_kv = math.fsum(plan.lanes[0].historical_kv_bytes for plan in token_zero)
    increment = math.fsum(plan.lanes[0].new_kv_bytes for plan in token_zero)
    side_capacity = initial_kv + 1.5 * increment

    result = decide_weight_residency(
        _architecture("b2_uhbm", 1e12, side_capacity),
        workload,
        _policy("streaming"),
    )

    assert not result.feasible
    assert result.reason is InfeasibilityReason.SIDE_KV_CAPACITY
    assert result.first_infeasible_token == 2


def test_policy_inputs_must_be_single_expanded_sweep_points():
    """Silently selecting the first value from an unexpanded sweep must fail."""
    from uhbm_simulator.prefetch_workload import decide_weight_residency

    policy = replace(
        _policy("streaming"),
        weight_residency_policy_values=("streaming", "persistent_if_fit"),
    )
    with pytest.raises(ValueError, match="expanded.*single"):
        decide_weight_residency(
            _architecture("b0_2_5d", 1e12), _workload(), policy
        )

