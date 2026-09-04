from dataclasses import replace

import pytest

from uhbm_simulator.simulator import SimulationParameters, run_simulation
from uhbm_simulator.workload import TrafficSignature


def parameters(comm_mode):
    return SimulationParameters(
        tp=2,
        r=0.0,
        r_remote=0.0,
        comm_mode=comm_mode,
        thermal_scenario="no_throttling",
        local_hbm_stack_count=1,
        local_hbm_stack_bw_bps=100.0,
        local_hbm_stack_capacity_bytes=1_000.0,
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
    )


def test_cut_through_keeps_source_read_but_bypasses_destination_hbm():
    signature = TrafficSignature("decode", 0.0, 100.0, 1, 0.0)

    materialized = run_simulation(signature, parameters("materialized"))
    cut_through = run_simulation(signature, parameters("cut_through"))

    materialized_local = sum(
        metric.transferred_amount
        for name, metric in materialized.resource_metrics.items()
        if name.startswith("local_hbm")
    )
    cut_through_local = sum(
        metric.transferred_amount
        for name, metric in cut_through.resource_metrics.items()
        if name.startswith("local_hbm")
    )

    assert materialized_local == 600.0
    assert cut_through_local == 200.0
    assert cut_through.communication_time_s == pytest.approx(2.0)
    assert cut_through.tpot_s < materialized.tpot_s


def test_cut_through_does_not_reserve_destination_hbm_capacity():
    signature = TrafficSignature("decode", 60.0, 50.0, 1, 0.0)

    result = run_simulation(
        signature,
        replace(
            parameters("cut_through"), local_hbm_stack_capacity_bytes=100.0
        ),
    )

    assert result.tpot_s > 0.0
