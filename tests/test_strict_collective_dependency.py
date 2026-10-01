from collections import defaultdict

import pytest

from uhbm_simulator.resources import RateResource, TrafficClass
from uhbm_simulator.simulator import SimulationParameters, run_simulation
from uhbm_simulator.workload import TrafficSignature


def test_strict_collective_dependency_defaults_to_true():
    parameters = SimulationParameters(
        tp=1,
        r=0.0,
        r_remote=0.0,
        comm_mode="materialized",
        thermal_scenario="no_throttling",
        local_hbm_stack_count=1,
        local_hbm_stack_bw_bps=1.0,
        local_hbm_stack_capacity_bytes=1.0,
        local_hbm_mapping_policy="uniform",
        local_hbm_mapping_weights=None,
        side_hbm_bw_bps=1.0,
        side_hbm_latency_s=0.0,
        h2h_bw_bps=1.0,
        h2h_latency_s=0.0,
        gpu_direct_bw_bps=1.0,
        gpu_direct_latency_s=0.0,
        gpu_peak_flops=1.0,
        base_die_latency_s=0.0,
    )

    assert parameters.strict_collective_dependency is True


def _run_strict_with_resource_trace(monkeypatch):
    starts = defaultdict(list)
    ends = defaultdict(list)
    original_use = RateResource.use

    def traced_use(
        resource, amount, traffic_class=TrafficClass.UNCLASSIFIED
    ):
        starts[resource].append(resource.env.now)
        yield from original_use(resource, amount, traffic_class)
        ends[resource].append(resource.env.now)

    monkeypatch.setattr(RateResource, "use", traced_use)
    signature = TrafficSignature(
        "decode",
        b_local_bytes=200.0,
        b_remote_bytes=40.0,
        n_messages=2,
        compute_flops=200.0,
        n_side_requests=0,
    )
    parameters = SimulationParameters(
        tp=2,
        r=1.0,
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
        h2h_latency_s=0.0,
        gpu_direct_bw_bps=10.0,
        gpu_direct_latency_s=0.0,
        gpu_peak_flops=100.0,
        base_die_latency_s=0.0,
        include_materialized_endpoints=False,
        allow_overlap=True,
        strict_collective_dependency=True,
    )

    result = run_simulation(signature, parameters)

    # run_simulation also executes a fresh sequential counterfactual to compute
    # overlap_s. The first resource instance with each name belongs to the
    # strict run; later instances with the same name belong to that fixture.
    def timeline(name):
        resource = next(resource for resource in starts if resource.name == name)
        return starts[resource], ends[resource]

    return result, timeline


def test_strict_dependency_delays_next_compute_until_previous_collective_finishes(
    monkeypatch,
):
    _, timeline = _run_strict_with_resource_trace(monkeypatch)

    compute_starts, _ = timeline("gpu_compute[0]")
    _, communication_ends = timeline("gpu_direct_link[0,1]")

    assert compute_starts[1] == pytest.approx(4.0)
    assert compute_starts[1] >= communication_ends[0]


def test_strict_dependency_still_prefetches_memory_during_previous_collective(
    monkeypatch,
):
    _, timeline = _run_strict_with_resource_trace(monkeypatch)

    memory_starts, _ = timeline("local_hbm[0][0]")
    _, communication_ends = timeline("gpu_direct_link[0,1]")

    assert memory_starts[1] == pytest.approx(2.0)
    assert memory_starts[1] < communication_ends[0]


def test_strict_dependency_joins_the_final_collective_before_tile_completion(
    monkeypatch,
):
    result, timeline = _run_strict_with_resource_trace(monkeypatch)

    _, communication_ends = timeline("gpu_direct_link[0,1]")

    assert communication_ends[-1] == pytest.approx(7.0)
    assert result.elapsed_s == pytest.approx(7.0)
    assert result.elapsed_s >= communication_ends[-1]
