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
    return replace(base, **overrides)


def signature(**overrides):
    values = dict(
        phase="decode",
        b_local_bytes=0.0,
        b_remote_bytes=100.0,
        n_messages=1,
        compute_flops=0.0,
    )
    values.update(overrides)
    return TrafficSignature(**values)


def test_materialized_uses_source_read_destination_write_and_reread():
    result = run_simulation(signature(), parameters())

    local_amount = sum(
        metric.transferred_amount
        for name, metric in result.resource_metrics.items()
        if name.startswith("local_hbm")
    )
    h2h_amount = sum(
        metric.transferred_amount
        for name, metric in result.resource_metrics.items()
        if name.startswith("h2h_link")
    )
    direct_amount = sum(
        metric.transferred_amount
        for name, metric in result.resource_metrics.items()
        if name.startswith("gpu_direct_link")
    )

    assert local_amount == 600.0  # two tiles x (source read + write + reread)
    assert h2h_amount == 200.0
    assert direct_amount == 0.0
    assert result.tpot_s == pytest.approx(4.0)


def test_r_one_routes_collective_only_over_gpu_direct_link():
    result = run_simulation(signature(), parameters(r=1.0))

    h2h_amount = sum(
        metric.transferred_amount
        for name, metric in result.resource_metrics.items()
        if name.startswith("h2h_link")
    )
    direct_amount = sum(
        metric.transferred_amount
        for name, metric in result.resource_metrics.items()
        if name.startswith("gpu_direct_link")
    )
    local_amount = sum(
        metric.transferred_amount
        for name, metric in result.resource_metrics.items()
        if name.startswith("local_hbm")
    )

    assert h2h_amount == 0.0
    assert direct_amount == 200.0
    assert local_amount == 0.0
    assert result.tpot_s == pytest.approx(1.0)


def test_rejects_working_set_that_exceeds_local_capacity():
    oversized = signature(b_local_bytes=200.0, b_remote_bytes=0.0, n_messages=0)

    with pytest.raises(ValueError, match="capacity"):
        run_simulation(
            oversized,
            parameters(local_hbm_stack_capacity_bytes=100.0, r_remote=0.25),
        )


def test_base_die_latency_occurs_after_h2h_transfer():
    result = run_simulation(signature(), parameters(base_die_latency_s=10.0))

    first_link = result.resource_metrics["h2h_link[0,1]"]
    assert first_link.first_service_start_s == pytest.approx(1.0)


def test_materialized_destination_buffer_counts_toward_hbm_capacity():
    buffered = signature(b_local_bytes=60.0, b_remote_bytes=50.0)

    with pytest.raises(ValueError, match="materialized"):
        run_simulation(buffered, parameters(local_hbm_stack_capacity_bytes=100.0))


def test_materialized_capacity_uses_peak_message_chunk_not_total_traffic():
    chunked = signature(b_local_bytes=60.0, b_remote_bytes=80.0, n_messages=4)

    result = run_simulation(
        chunked, parameters(local_hbm_stack_capacity_bytes=100.0)
    )

    assert result.tpot_s > 0.0


def test_endpoint_disabled_validation_does_not_reserve_receive_buffer():
    validation = signature(b_local_bytes=60.0, b_remote_bytes=100.0)

    result = run_simulation(
        validation,
        parameters(
            local_hbm_stack_capacity_bytes=60.0,
            include_materialized_endpoints=False,
        ),
    )

    assert result.tpot_s > 0.0
