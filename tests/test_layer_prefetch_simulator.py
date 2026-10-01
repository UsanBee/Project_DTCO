import math

import pytest

from uhbm_simulator.prefetch_config import (
    ArchitectureConfig,
    CapacityMemoryConfig,
    CapacityMemoryDeviceConfig,
    LinkConfig,
    MemoryDeviceConfig,
    PrefetchExecutionConfig,
    PrefetchPolicyConfig,
    PrefetchSimulationConfig,
    PrefetchWorkloadConfig,
)
from uhbm_simulator.resources import TrafficClass


def _config(
    kind="b2_uhbm",
    *,
    capacity_bw=1e15,
    local_bw=1e15,
    side_bw=1e15,
    gpu_flops=1e9,
    run_mode="decode_only",
    start_state="cold_start",
    decode_tokens=1,
):
    local = MemoryDeviceConfig(1e15, local_bw, 0.0, 1.0)
    side = (
        MemoryDeviceConfig(1e15, side_bw, 0.0, 1.0)
        if kind == "b2_uhbm"
        else None
    )
    side_link = LinkConfig(side_bw, 0.0, 1.0) if side is not None else None
    architecture = ArchitectureConfig(
        name=kind,
        kind=kind,
        lane_count=8,
        local_memory=local,
        side_hbm=side,
        side_link=side_link,
    )
    return PrefetchSimulationConfig(
        mode="layer_prefetch",
        architectures=(architecture,),
        capacity_memory=CapacityMemoryConfig(
            CapacityMemoryDeviceConfig(1e15, capacity_bw, 0.0, 1.0),
            LinkConfig(capacity_bw, 0.0, 1.0, 20.0),
        ),
        execution=PrefetchExecutionConfig(
            gpu_peak_flops=gpu_flops,
            r=0.0,
            comm_mode="cut_through",
            h2h_bw_bps=1e15,
            h2h_latency_s=0.0,
            gpu_direct_bw_bps=1e15,
            gpu_direct_latency_s=0.0,
            base_die_latency_s=0.0,
            include_materialized_endpoints=False,
        ),
        prefetch=PrefetchPolicyConfig(
            weight_residency_policy_values=("streaming",),
            weight_chunk_bytes_values=(10**18,),
            kv_chunk_bytes_values=(10**18,),
            kv_writeback_chunk_bytes_values=(10**18,),
            max_inflight_chunks_per_lane_values=(None,),
        ),
        workload=PrefetchWorkloadConfig(
            run_mode_values=(run_mode,),
            start_state_values=(start_state,),
            model="Llama-7B",
            tp=2,
            pp=1,
            seq_len=64,
            batch_values=(1,),
            decode_tokens_values=(decode_tokens,),
        ),
    )


def test_next_layer_prefetch_launches_at_current_layer_start_and_waits_all_lanes():
    """Late launch or consuming a partial lane arrival must fail this trace."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    result = run_prefetch_simulation(_config(), detailed_trace=True)
    first, second = result.trace.layer_timings[:2]

    assert second.prefetch_launch_s == first.compute_start_s
    assert first.layer_ready_s == max(
        max(first.weight_lane_ready_s), max(first.kv_lane_ready_s)
    )
    assert first.compute_start_s == max(first.lane_ready_s)
    assert second.compute_start_s >= max(second.lane_ready_s)
    assert len(first.lane_ready_s) == 8


def test_slowest_lane_ready_time_is_the_full_layer_barrier():
    """Using mean lane readiness instead of max would permit partial consume."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    result = run_prefetch_simulation(_config(), detailed_trace=True)
    for timing in result.trace.layer_timings:
        assert timing.layer_ready_s == max(timing.lane_ready_s)
        assert timing.compute_start_s >= timing.layer_ready_s


def test_positive_prefetch_slack_creates_no_stall():
    """Charging hidden fast prefetch as exposed latency must fail."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    result = run_prefetch_simulation(
        _config(capacity_bw=1e15, gpu_flops=1e8), detailed_trace=True
    )
    hidden = [
        timing
        for timing in result.trace.layer_timings[1:]
        if timing.prefetch_slack_s > 0.0
    ]

    assert hidden
    assert all(timing.prefetch_stall_s == 0.0 for timing in hidden)


def test_negative_prefetch_slack_adds_exactly_its_magnitude():
    """A heuristic stall correction instead of event timing must fail."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    result = run_prefetch_simulation(
        _config(capacity_bw=1e7, gpu_flops=1e15), detailed_trace=True
    )
    exposed = [
        timing
        for timing in result.trace.layer_timings[1:]
        if timing.prefetch_slack_s < 0.0
    ]

    assert exposed
    for timing in exposed:
        assert timing.prefetch_stall_s == pytest.approx(-timing.prefetch_slack_s)
        assert timing.compute_start_s - timing.previous_layer_end_s == pytest.approx(
            timing.prefetch_stall_s
        )


def test_b0_and_b2_use_their_distinct_prefetch_paths():
    """Routing B0 KV or B2 weight through the wrong tier must fail."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    b0 = run_prefetch_simulation(_config(kind="b0_2_5d"), detailed_trace=True)
    b2 = run_prefetch_simulation(_config(kind="b2_uhbm"), detailed_trace=True)

    assert b0.path_bytes[TrafficClass.CAPACITY_TO_2_5D_HBM] > 0.0
    assert b0.path_bytes.get(TrafficClass.SIDE_TO_UHBM, 0.0) == 0.0
    assert b2.path_bytes[TrafficClass.CAPACITY_TO_UHBM] > 0.0
    assert b2.path_bytes[TrafficClass.SIDE_TO_UHBM] > 0.0
    assert b2.path_bytes[TrafficClass.UHBM_TO_SIDE] > 0.0


def test_attention_mlp_and_next_layer_keep_strict_collective_dependency():
    """Reintroducing the relaxed causal DAG must fail these ordering checks."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    result = run_prefetch_simulation(_config(), detailed_trace=True)
    timings = result.trace.layer_timings

    for timing in timings:
        assert timing.mlp_compute_start_s >= timing.attention_collective_end_s
        assert timing.mlp_collective_end_s == timing.compute_end_s
    for previous, current in zip(timings, timings[1:]):
        assert current.compute_start_s >= previous.mlp_collective_end_s


def test_path_byte_accounting_is_finite_and_nonnegative():
    """NaN or negative path accounting would invalidate later energy metrics."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    result = run_prefetch_simulation(_config(), detailed_trace=True)
    assert result.path_bytes
    assert all(math.isfinite(value) and value >= 0.0 for value in result.path_bytes.values())
