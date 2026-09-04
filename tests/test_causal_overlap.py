from dataclasses import replace

import pytest

from uhbm_simulator.simulator import SimulationParameters, run_simulation
from uhbm_simulator.workload import TrafficSignature


def parameters(**overrides):
    base = SimulationParameters(
        tp=2,
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
        h2h_latency_s=0.0,
        gpu_direct_bw_bps=100.0,
        gpu_direct_latency_s=0.0,
        gpu_peak_flops=100.0,
        base_die_latency_s=0.0,
        include_materialized_endpoints=False,
        allow_overlap=True,
    )
    return replace(base, **overrides)


def test_single_stage_obeys_memory_compute_communication_causality():
    signature = TrafficSignature(
        "decode", 100.0, 100.0, 1, 100.0, n_side_requests=0
    )

    result = run_simulation(signature, parameters())

    assert result.tpot_s == pytest.approx(3.0)
    assert result.overlap_s == 0.0


def test_later_stage_work_can_overlap_only_after_first_producer_completes():
    signature = TrafficSignature(
        "decode", 100.0, 100.0, 2, 100.0, n_side_requests=0
    )

    result = run_simulation(signature, parameters())

    assert result.tpot_s == pytest.approx(2.5)
    assert result.overlap_s == pytest.approx(0.5)
    assert result.communication_time_s == pytest.approx(1.0)


def test_side_hbm_latency_is_paid_n_side_times():
    signature = TrafficSignature(
        "decode",
        b_local_bytes=100.0,
        b_remote_bytes=0.0,
        n_messages=0,
        compute_flops=0.0,
        n_side_requests=4,
    )

    result = run_simulation(
        signature,
        parameters(tp=1, r_remote=1.0, side_hbm_latency_s=1.0),
    )

    assert result.tpot_s == pytest.approx(5.0)
    assert result.resource_metrics["side_hbm[0]"].request_count == 4


@pytest.mark.parametrize(
    "signature",
    [
        TrafficSignature("decode", 100.0, 0.0, 0, 0.0, n_side_requests=0),
        TrafficSignature("decode", 0.0, 100.0, 0, 0.0, n_side_requests=0),
    ],
)
def test_rejects_nonzero_traffic_with_zero_request_count(signature):
    kwargs = {"r_remote": 1.0} if signature.b_local_bytes else {}

    with pytest.raises(ValueError, match="request"):
        run_simulation(signature, parameters(**kwargs))
