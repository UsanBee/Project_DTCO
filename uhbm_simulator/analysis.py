"""Stage 1/Stage 2 comparison helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from stage1_model import (
    BW_INTERTILE_0,
    BW_LOCAL_0,
    K_L,
    L_DIRECT_COLLECTIVE_0,
    L_H2H_ROUND_0,
    ring_allreduce_round_count,
)

from uhbm_simulator.simulator import SimulationParameters, run_simulation
from uhbm_simulator.workload import build_decode_signature


@dataclass(frozen=True)
class H2HFeasibility:
    required_bw_bps: float
    required_latency_s: float
    bandwidth_feasible: bool
    latency_feasible: bool
    architecture_feasible: bool


def _stage1_baseline_time_s(signature) -> float:
    return (
        signature.b_local_bytes / BW_LOCAL_0
        + signature.n_messages * L_DIRECT_COLLECTIVE_0
        + signature.b_remote_bytes / BW_INTERTILE_0
    )


def _equivalence_time_s(signature, tp: int, k_b: float, h2h_bw_bps: float) -> float:
    parameters = SimulationParameters(
        tp=tp,
        r=0.0,
        r_remote=0.0,
        comm_mode="materialized",
        thermal_scenario="no_throttling",
        local_hbm_stack_count=1,
        local_hbm_stack_bw_bps=k_b * BW_LOCAL_0,
        local_hbm_stack_capacity_bytes=math.inf,
        local_hbm_mapping_policy="uniform",
        local_hbm_mapping_weights=None,
        side_hbm_bw_bps=BW_LOCAL_0,
        side_hbm_latency_s=0.0,
        h2h_bw_bps=h2h_bw_bps,
        h2h_latency_s=K_L * L_H2H_ROUND_0,
        gpu_direct_bw_bps=BW_INTERTILE_0,
        gpu_direct_latency_s=L_DIRECT_COLLECTIVE_0,
        gpu_peak_flops=math.inf,
        base_die_latency_s=0.0,
        include_materialized_endpoints=False,
        allow_overlap=False,
    )
    return run_simulation(signature, parameters).tpot_s


def empirical_break_even_bw(
    *,
    model_name: str,
    tp: int,
    seq_len: int,
    k_b: float,
    batch: int = 1,
    relative_tolerance: float = 1e-8,
) -> float:
    """Find the finite H2H bandwidth where Stage 2 crosses Stage 1.

    This validation fixture fixes r=0 and r_remote=0, disables overlap and
    endpoint costs, and removes queue contention. Only BW_H2H is swept.
    """
    if k_b <= 0:
        raise ValueError("k_b must be positive")
    built_signature = build_decode_signature(model_name, tp, seq_len, batch)
    signature = replace(
        built_signature,
        compute_flops=0.0,
        semantic_stages=tuple(
            replace(stage, compute_flops=0.0)
            for stage in built_signature.semantic_stages or ()
        ),
    )
    if signature.b_remote_bytes == 0:
        return 0.0

    baseline_time = _stage1_baseline_time_s(signature)

    def difference(bandwidth: float) -> float:
        return _equivalence_time_s(signature, tp, k_b, bandwidth) - baseline_time

    pivot = 1e6
    pivot_difference = difference(pivot)
    if pivot_difference > 0:
        low = pivot
        high = pivot
        high_difference = pivot_difference
        while high_difference > 0 and high < 1e18:
            high *= 10.0
            high_difference = difference(high)
    else:
        high = pivot
        high_difference = pivot_difference
        low = pivot / 10.0
        low_difference = difference(low)
        while low_difference <= 0 and low > 1e-300:
            high = low
            high_difference = low_difference
            low /= 10.0
            low_difference = difference(low)
    if high_difference > 0:
        return math.inf

    while high / low - 1.0 > relative_tolerance:
        midpoint = math.sqrt(low * high)
        if difference(midpoint) > 0:
            low = midpoint
        else:
            high = midpoint
    return math.sqrt(low * high)


def h2h_feasibility(
    *,
    model_name: str,
    tp: int,
    seq_len: int,
    k_b: float,
    batch: int,
    actual_bw_bps: float,
    actual_h2h_round_latency_s: float,
    signature=None,
    direct_collective_latency_s: float = L_DIRECT_COLLECTIVE_0,
    nominal_h2h_round_latency_s: float | None = None,
) -> H2HFeasibility:
    """Evaluate r=0 H2H constraints using per-round latency semantics.

    ``required_latency_s`` remains the compatibility field name, but its value
    is the maximum allowed H2H latency for one ring round.
    """
    if actual_bw_bps <= 0 or actual_h2h_round_latency_s < 0:
        raise ValueError("actual H2H bandwidth must be positive and latency nonnegative")
    if direct_collective_latency_s < 0.0:
        raise ValueError("direct_collective_latency_s cannot be negative")
    if nominal_h2h_round_latency_s is None:
        nominal_h2h_round_latency_s = K_L * L_H2H_ROUND_0
    if nominal_h2h_round_latency_s < 0.0:
        raise ValueError("nominal_h2h_round_latency_s cannot be negative")
    if signature is None:
        signature = build_decode_signature(model_name, tp, seq_len, batch)
    if signature.b_remote_bytes == 0:
        return H2HFeasibility(0.0, math.inf, True, True, True)

    memory_time_0 = signature.b_local_bytes / BW_LOCAL_0
    communication_time_0 = (
        signature.n_messages * direct_collective_latency_s
        + signature.b_remote_bytes / BW_INTERTILE_0
    )
    total_budget = communication_time_0 + memory_time_0 * (1.0 - 1.0 / k_b)
    n_rounds = ring_allreduce_round_count(tp)
    bandwidth_budget = (
        total_budget
        - signature.n_messages * n_rounds * nominal_h2h_round_latency_s
    )
    required_bw = (
        signature.b_remote_bytes / bandwidth_budget
        if bandwidth_budget > 0.0
        else math.inf
    )
    required_latency = (
        total_budget - signature.b_remote_bytes / actual_bw_bps
    ) / (signature.n_messages * n_rounds)
    bandwidth_feasible = actual_bw_bps >= required_bw
    latency_feasible = (
        required_latency >= 0.0
        and actual_h2h_round_latency_s <= required_latency
    )
    return H2HFeasibility(
        required_bw_bps=required_bw,
        required_latency_s=required_latency,
        bandwidth_feasible=bandwidth_feasible,
        latency_feasible=latency_feasible,
        architecture_feasible=bandwidth_feasible and latency_feasible,
    )
