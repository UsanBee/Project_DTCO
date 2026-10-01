"""Regression coverage for the production strict TP dependency invariant.

The relaxed process is an ablation path and is intentionally not required to
satisfy this invariant.
"""

import simpy

import uhbm_simulator.simulator as simulator_module
from uhbm_simulator.resources import build_resource_layer
from uhbm_simulator.simulator import (
    H2HTrafficCounters,
    RingCollectiveCoordinator,
    SimulationParameters,
    _communication_message,
    _strict_causal_tile_process,
)
from uhbm_simulator.workload import build_prefill_signature, split_traffic


EPSILON_S = 1e-12


def _instrument_strict_tile_zero(
    monkeypatch,
    env,
    streams,
    signature,
    parameters,
    resources,
    communication_intervals,
    coordinator,
    counters,
):
    """Run the production strict process while recording tile-0 boundaries."""
    compute_start_times = []
    collective_end_times = []

    original_compute_use = resources.gpu_compute[0].use

    def traced_compute_use(amount):
        compute_start_times.append(env.now)
        yield from original_compute_use(amount)

    def traced_communication_message(
        message_env,
        source,
        destination,
        h2h_amount,
        direct_amount,
        message_parameters,
        message_resources,
        *,
        coordinator=None,
        collective_index=0,
    ):
        yield message_env.process(
            _communication_message(
                message_env,
                source,
                destination,
                h2h_amount,
                direct_amount,
                message_parameters,
                message_resources,
                coordinator=coordinator,
                collective_index=collective_index,
            )
        )
        if source == 0:
            collective_end_times.append(message_env.now)

    monkeypatch.setattr(resources.gpu_compute[0], "use", traced_compute_use)
    monkeypatch.setattr(
        simulator_module,
        "_communication_message",
        traced_communication_message,
    )
    process = env.process(
        _strict_causal_tile_process(
            env,
            0,
            streams,
            signature,
            parameters,
            resources,
            communication_intervals,
            counters,
            coordinator,
        )
    )
    return process, compute_start_times, collective_end_times


def test_strict_mode_compute_never_starts_before_prior_collective_completes(
    monkeypatch,
):
    signature = build_prefill_signature(
        "Llama-70B",
        tp=4,
        seq_len=4096,
        batch=1,
    )
    parameters = SimulationParameters(
        tp=4,
        r=0.0,
        r_remote=0.1,
        comm_mode="materialized",
        thermal_scenario="no_throttling",
        local_hbm_stack_count=4,
        local_hbm_stack_bw_bps=1.2e12,
        local_hbm_stack_capacity_bytes=16e9,
        local_hbm_mapping_policy="uniform",
        local_hbm_mapping_weights=None,
        side_hbm_bw_bps=4.8e12,
        side_hbm_latency_s=50e-9,
        h2h_bw_bps=336e9,
        h2h_latency_s=20e-9,
        gpu_direct_bw_bps=768e9,
        gpu_direct_latency_s=20e-9,
        gpu_peak_flops=990e12,
        base_die_latency_s=5e-9,
        include_materialized_endpoints=True,
        allow_overlap=True,
        strict_collective_dependency=True,
    )
    streams = split_traffic(signature, parameters.r_remote, parameters.r)
    env = simpy.Environment()
    resources = build_resource_layer(
        env=env,
        tp=parameters.tp,
        r=parameters.r,
        local_hbm_stack_count=parameters.local_hbm_stack_count,
        local_hbm_stack_bw_bps=parameters.local_hbm_stack_bw_bps,
        local_hbm_stack_capacity_bytes=parameters.local_hbm_stack_capacity_bytes,
        local_hbm_latency_s=parameters.local_hbm_latency_s,
        side_hbm_bw_bps=parameters.side_hbm_bw_bps,
        side_hbm_latency_s=parameters.side_hbm_latency_s,
        h2h_bw_bps=parameters.h2h_bw_bps,
        h2h_latency_s=parameters.h2h_latency_s,
        gpu_direct_bw_bps=parameters.gpu_direct_bw_bps,
        gpu_direct_latency_s=parameters.gpu_direct_latency_s,
        gpu_compute_flops=parameters.gpu_peak_flops,
    )
    communication_intervals = {}
    counters = H2HTrafficCounters()
    coordinator = RingCollectiveCoordinator(
        env,
        parameters,
        resources,
        counters,
    )

    tile_zero, compute_starts, collective_ends = _instrument_strict_tile_zero(
        monkeypatch,
        env,
        streams,
        signature,
        parameters,
        resources,
        communication_intervals,
        coordinator,
        counters,
    )
    other_tiles = [
        env.process(
            _strict_causal_tile_process(
                env,
                tile,
                streams,
                signature,
                parameters,
                resources,
                communication_intervals,
                counters,
                coordinator,
            )
        )
        for tile in range(1, parameters.tp)
    ]
    env.run(until=env.all_of([tile_zero, *other_tiles]))

    assert signature.n_messages == 160
    assert len(compute_starts) == signature.n_messages
    assert len(collective_ends) == signature.n_messages
    for previous_stage in range(signature.n_messages - 1):
        assert (
            compute_starts[previous_stage + 1]
            >= collective_ends[previous_stage] - EPSILON_S
        )
