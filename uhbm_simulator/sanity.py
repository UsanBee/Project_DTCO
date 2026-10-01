"""Executable sanity checks from spec chapter 10."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from stage1_model import (
    BW_INTERTILE_0,
    BW_LOCAL_0,
    L_D2D_0,
    MODELS,
    required_bw_d2d,
)

from uhbm_simulator.analysis import empirical_break_even_bw
from uhbm_simulator.config import DEFAULT_H2H_LANE_COUNT, h2h_bandwidth_bps
from uhbm_simulator.simulator import SimulationParameters, run_simulation
from uhbm_simulator.workload import TrafficSignature, build_decode_signature


@dataclass(frozen=True)
class SanityReport:
    checks: dict[str, bool]
    details: dict[str, float]


def _parameters(**overrides) -> SimulationParameters:
    base = SimulationParameters(
        tp=2,
        r=0.0,
        r_remote=0.0,
        comm_mode="materialized",
        thermal_scenario="no_throttling",
        local_hbm_stack_count=1,
        local_hbm_stack_bw_bps=100.0,
        local_hbm_stack_capacity_bytes=math.inf,
        local_hbm_mapping_policy="uniform",
        local_hbm_mapping_weights=None,
        side_hbm_bw_bps=100.0,
        side_hbm_latency_s=0.0,
        h2h_bw_bps=100.0,
        h2h_latency_s=0.0,
        gpu_direct_bw_bps=100.0,
        gpu_direct_latency_s=0.0,
        gpu_peak_flops=100.0,
        base_die_latency_s=0.0,
        include_materialized_endpoints=False,
        allow_overlap=False,
    )
    return replace(base, **overrides)


def _total_amount(result, prefix: str) -> float:
    return sum(
        metric.transferred_amount
        for name, metric in result.resource_metrics.items()
        if name.startswith(prefix)
    )


def run_sanity_checks() -> SanityReport:
    """Run every monotonicity and equivalence check required by the spec."""
    expected_break_even = required_bw_d2d(
        MODELS["Llama-70B"], tp=4, seq_len=4096, k_B=2.0
    )
    empirical_break_even = empirical_break_even_bw(
        model_name="Llama-70B", tp=4, seq_len=4096, k_b=2.0
    )
    break_even_error = abs(empirical_break_even - expected_break_even) / expected_break_even

    local_signature = TrafficSignature("decode", 100.0, 0.0, 0, 0.0)
    low_local_bw = run_simulation(
        local_signature, _parameters(tp=1, local_hbm_stack_bw_bps=10.0)
    )
    high_local_bw = run_simulation(
        local_signature, _parameters(tp=1, local_hbm_stack_bw_bps=20.0)
    )

    comm_signature = TrafficSignature("decode", 0.0, 100.0, 1, 0.0)
    low_h2h_bw = run_simulation(
        comm_signature, _parameters(h2h_bw_bps=10.0)
    )
    high_h2h_bw = run_simulation(
        comm_signature, _parameters(h2h_bw_bps=20.0)
    )

    decode_signature = TrafficSignature("decode", 0.0, 100.0, 10, 0.0)
    prefill_signature = TrafficSignature("prefill", 0.0, 100.0, 1, 0.0)
    decode_low_latency = run_simulation(
        decode_signature, _parameters(h2h_bw_bps=1e12, h2h_latency_s=0.0)
    )
    decode_high_latency = run_simulation(
        decode_signature, _parameters(h2h_bw_bps=1e12, h2h_latency_s=1.0)
    )
    prefill_low_latency = run_simulation(
        prefill_signature, _parameters(h2h_bw_bps=1e12, h2h_latency_s=0.0)
    )
    prefill_high_latency = run_simulation(
        prefill_signature, _parameters(h2h_bw_bps=1e12, h2h_latency_s=1.0)
    )
    decode_latency_delta = (
        decode_high_latency.communication_time_s
        - decode_low_latency.communication_time_s
    )
    prefill_latency_delta = (
        prefill_high_latency.communication_time_s
        - prefill_low_latency.communication_time_s
    )

    r_results = [
        run_simulation(comm_signature, _parameters(r=r)) for r in (0.0, 0.5, 1.0)
    ]
    h2h_amounts = [_total_amount(result, "h2h_link") for result in r_results]

    built_stage1_signature = build_decode_signature(
        "Llama-70B", tp=4, seq_len=4096
    )
    stage1_signature = replace(
        built_stage1_signature,
        compute_flops=0.0,
        semantic_stages=tuple(
            replace(stage, compute_flops=0.0)
            for stage in built_stage1_signature.semantic_stages or ()
        ),
    )
    r_one = run_simulation(
        stage1_signature,
        _parameters(
            tp=4,
            r=1.0,
            local_hbm_stack_bw_bps=BW_LOCAL_0,
            gpu_direct_bw_bps=BW_INTERTILE_0,
            gpu_direct_latency_s=L_D2D_0,
        ),
    )
    conventional_comm = (
        stage1_signature.n_messages * L_D2D_0
        + stage1_signature.b_remote_bytes / BW_INTERTILE_0
    )

    pillar_bandwidths = [
        h2h_bandwidth_bps(48.0, pillar, 64, DEFAULT_H2H_LANE_COUNT)
        for pillar in (4, 8, 16, 32)
    ]

    compute_signature = TrafficSignature("prefill", 0.0, 0.0, 0, 100.0)
    thermal_times = [
        run_simulation(
            compute_signature,
            _parameters(tp=1, thermal_scenario=scenario),
        ).tpot_s
        for scenario in (
            "no_throttling",
            "mild_throttling",
            "severe_throttling",
        )
    ]

    checks = {
        "stage1_break_even_matches": break_even_error < 1e-6,
        "higher_uhbm_bw_lowers_tpot": high_local_bw.tpot_s < low_local_bw.tpot_s,
        "higher_h2h_bw_lowers_comm_time": (
            high_h2h_bw.communication_time_s < low_h2h_bw.communication_time_s
        ),
        "decode_is_more_latency_sensitive": (
            decode_latency_delta > prefill_latency_delta
        ),
        "higher_r_reduces_h2h_dependency": all(
            left > right for left, right in zip(h2h_amounts, h2h_amounts[1:])
        ),
        "r_one_matches_conventional_b0": math.isclose(
            r_one.communication_time_s, conventional_comm, rel_tol=1e-10
        ),
        "higher_n_pillar_lowers_h2h_bw": all(
            left > right
            for left, right in zip(pillar_bandwidths, pillar_bandwidths[1:])
        ),
        "thermal_scenarios_order_compute_time": (
            thermal_times[0] < thermal_times[1] < thermal_times[2]
        ),
    }
    return SanityReport(
        checks=checks,
        details={
            "stage1_break_even_bps": expected_break_even,
            "stage2_break_even_bps": empirical_break_even,
            "stage1_relative_error": break_even_error,
            "decode_latency_delta_s": decode_latency_delta,
            "prefill_latency_delta_s": prefill_latency_delta,
        },
    )
