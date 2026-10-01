import math

import pytest
import simpy

from uhbm_simulator.resources import build_resource_layer
from uhbm_simulator.simulator import (
    H2HTrafficCounters,
    RingCollectiveCoordinator,
    SimulationParameters,
    _split_collective_round_bytes,
    run_simulation,
)
from uhbm_simulator.workload import TrafficSignature


def _parameters(**overrides):
    values = dict(
        tp=4,
        r=0.0,
        r_remote=0.0,
        comm_mode="materialized",
        thermal_scenario="no_throttling",
        local_hbm_stack_count=1,
        local_hbm_stack_bw_bps=100.0,
        local_hbm_stack_capacity_bytes=10_000.0,
        local_hbm_mapping_policy="uniform",
        local_hbm_mapping_weights=None,
        side_hbm_bw_bps=100.0,
        side_hbm_latency_s=0.0,
        h2h_bw_bps=100.0,
        h2h_latency_s=1.0,
        gpu_direct_bw_bps=100.0,
        gpu_direct_latency_s=0.0,
        gpu_peak_flops=100.0,
        base_die_latency_s=0.0,
        include_materialized_endpoints=False,
        allow_overlap=False,
    )
    values.update(overrides)
    return SimulationParameters(**values)


def _resources(env, parameters):
    return build_resource_layer(
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


def test_round_payload_split_conserves_the_aggregate_with_math_fsum():
    rounds = _split_collective_round_bytes(10.0, tp=4)

    assert len(rounds) == 6
    assert math.fsum(rounds) == 10.0


def test_tp4_collective_uses_six_rounds_without_multiplying_aggregate_bytes():
    signature = TrafficSignature("decode", 0.0, 120.0, 1, 0.0)
    result = run_simulation(signature, _parameters())

    links = [
        metric
        for name, metric in result.resource_metrics.items()
        if name.startswith("h2h_link")
    ]
    assert len(links) == 4
    assert all(metric.request_count == 6 for metric in links)
    assert all(metric.transferred_amount == 120.0 for metric in links)
    assert all(metric.total_latency_time_s == 6.0 for metric in links)
    assert result.collective_h2h_edge_bytes == 480.0
    assert result.side_hbm_read_h2h_edge_bytes == 0.0
    # Each 20-byte round takes 0.2 s service + 1.0 s fixed latency, and the
    # coordinator's global barrier serializes all six rounds.
    assert result.elapsed_s == pytest.approx(6 * (20.0 / 100.0 + 1.0))


def test_materialized_collective_final_result_is_written_and_reread_from_requester_local_hbm():
    parameters = _parameters(
        h2h_latency_s=0.0,
        include_materialized_endpoints=True,
    )
    env = simpy.Environment()
    resources = _resources(env, parameters)
    counters = H2HTrafficCounters()
    coordinator = RingCollectiveCoordinator(
        env, parameters, resources, counters
    )

    # Distinct per-rank payloads make every rank's self-materialization
    # distinguishable from the retired next-tile target.
    per_rank_amounts = (10.0, 20.0, 30.0, 40.0)
    processes = [
        env.process(
            coordinator.participate(
                collective_index=0,
                requester=tile,
                h2h_collective_bytes=per_rank_amounts[tile],
                gpu_direct_collective_bytes=0.0,
            )
        )
        for tile in range(4)
    ]
    env.run(until=env.all_of(processes))

    for tile, amount in enumerate(per_rank_amounts):
        endpoint = resources.local_hbm[(tile, 0)].stats
        assert endpoint.transferred_amount == 3.0 * amount
        assert endpoint.request_count == 3


def test_hybrid_collective_completion_waits_for_both_route_abstractions():
    signature = TrafficSignature("decode", 0.0, 120.0, 1, 0.0)
    result = run_simulation(
        signature,
        _parameters(
            r=0.5,
            h2h_bw_bps=1e9,
            h2h_latency_s=1.0,
            gpu_direct_bw_bps=1.0,
            gpu_direct_latency_s=0.0,
        ),
    )

    # H2H takes about six seconds of round latency, while the topology-abstract
    # direct route takes 60 seconds. Completion must wait for the slower route.
    assert result.elapsed_s == pytest.approx(60.0)


def test_direct_only_collective_preserves_single_aggregate_event_timing():
    signature = TrafficSignature("decode", 0.0, 120.0, 1, 0.0)
    result = run_simulation(
        signature,
        _parameters(
            r=1.0,
            gpu_direct_bw_bps=100.0,
            gpu_direct_latency_s=0.5,
        ),
    )

    assert result.elapsed_s == pytest.approx(120.0 / 100.0 + 0.5)
    assert result.collective_h2h_edge_bytes == 0.0
    assert all(
        metric.request_count == 1
        for name, metric in result.resource_metrics.items()
        if name.startswith("gpu_direct_link")
    )
