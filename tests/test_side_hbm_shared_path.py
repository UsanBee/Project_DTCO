from dataclasses import replace

import pytest

from uhbm_simulator.config import load_config
from uhbm_simulator.simulator import SimulationParameters, run_simulation
from uhbm_simulator.sweep import run_sweep
from uhbm_simulator.workload import TrafficSignature


def _parameters(**overrides):
    base = SimulationParameters(
        tp=4,
        r=0.0,
        r_remote=1.0,
        comm_mode="cut_through",
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
        allow_overlap=False,
    )
    return replace(base, **overrides)


def test_shared_side_hbm_remote_read_uses_owner_to_requester_ring_edges():
    signature = TrafficSignature(
        "decode", 100.0, 0.0, 0, 0.0, n_side_requests=1
    )
    result = run_simulation(signature, _parameters())

    # Request i=0 places tile t's data at owner t+1. The large read payload
    # returns owner->requester, which is three edges on a directed TP=4 ring.
    assert result.side_hbm_read_h2h_edge_bytes == 1_200.0
    assert result.collective_h2h_edge_bytes == 0.0
    assert sum(
        metric.transferred_amount
        for name, metric in result.resource_metrics.items()
        if name.startswith("side_hbm")
    ) == 400.0
    assert all(
        result.resource_metrics[f"h2h_link[{edge[0]},{edge[1]}]"].transferred_amount
        == 300.0
        for edge in ((0, 1), (1, 2), (2, 3), (3, 0))
    )


def test_independent_side_hbm_path_preserves_the_legacy_no_h2h_behavior():
    signature = TrafficSignature(
        "decode", 100.0, 0.0, 0, 0.0, n_side_requests=1
    )
    result = run_simulation(
        signature,
        _parameters(side_hbm_path="independent"),
    )

    assert result.side_hbm_read_h2h_edge_bytes == 0.0
    assert sum(
        metric.transferred_amount
        for name, metric in result.resource_metrics.items()
        if name.startswith("h2h_link")
    ) == 0.0


def test_tp1_shared_side_path_is_bit_exact_with_independent_path():
    signature = TrafficSignature(
        "decode", 100.0, 0.0, 0, 0.0, n_side_requests=1
    )

    shared = run_simulation(signature, _parameters(tp=1))
    independent = run_simulation(
        signature,
        _parameters(tp=1, side_hbm_path="independent"),
    )

    assert shared == independent


def test_shared_side_and_collective_bytes_use_the_same_h2h_resources():
    signature = TrafficSignature(
        "decode",
        b_local_bytes=200.0,
        b_remote_bytes=240.0,
        n_messages=2,
        compute_flops=0.0,
        n_side_requests=2,
        local_footprint_bytes=200.0,
    )
    result = run_simulation(
        signature,
        _parameters(h2h_bw_bps=100.0, allow_overlap=True),
    )

    edge_metrics = [
        metric
        for name, metric in result.resource_metrics.items()
        if name.startswith("h2h_link")
    ]
    assert sum(metric.transferred_amount for metric in edge_metrics) == pytest.approx(
        result.collective_h2h_edge_bytes
        + result.side_hbm_read_h2h_edge_bytes
    )
    # Stage-1 side reads overlap the previous strict collective and contend on
    # the shared physical edge resources rather than bypassing them.
    assert sum(metric.total_queue_time_s for metric in edge_metrics) > 0.0


def test_shared_path_moves_side_link_energy_into_h2h_energy(tmp_path):
    config = load_config("configs/stage2_default.yaml")
    reduced = replace(
        config,
        r_values=(1.0,),
        side_hbm_path_values=("shared_h2h", "independent"),
        channels=replace(
            config.channels,
            h2h=replace(
                config.channels.h2h,
                lane_rate_gbps=(12.0,),
                n_pillar=(4,),
            ),
        ),
        workload=replace(
            config.workload,
            model="Llama-7B",
            seq_len=128,
            batch_values=(1,),
            r_remote_values=(0.1,),
        ),
    )
    rows = {row.side_hbm_path: row for row in run_sweep(reduced)}

    assert rows["shared_h2h"].side_hbm_link_energy_j == 0.0
    assert rows["shared_h2h"].h2h_energy_j > 0.0
    assert rows["shared_h2h"].architecture_feasibility_applicable is False
    assert rows["independent"].side_hbm_link_energy_j > 0.0
    assert rows["independent"].side_hbm_read_h2h_edge_bytes == 0.0
    assert rows["independent"].architecture_feasibility_applicable is True


def test_unknown_side_hbm_path_is_rejected():
    signature = TrafficSignature("decode", 1.0, 0.0, 0, 0.0)

    with pytest.raises(ValueError, match="side_hbm_path"):
        run_simulation(signature, _parameters(side_hbm_path="unknown"))
