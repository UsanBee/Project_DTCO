from dataclasses import replace
import math
from pathlib import Path

import pytest
import simpy
import yaml

from uhbm_simulator.config import load_config
from uhbm_simulator.resources import build_resource_layer
from uhbm_simulator.simulator import (
    SimulationParameters,
    _side_hbm_owner,
    calculate_channel_energy,
    run_simulation,
)
from uhbm_simulator.workload import TrafficSignature
from uhbm_simulator.sweep import run_sweep


ADJACENCY_TP4 = ((1,), (0,), (3,), (2,))


def _parameters(**overrides):
    base = SimulationParameters(
        tp=4,
        r=0.0,
        r_remote=1.0,
        comm_mode="cut_through",
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
        allow_overlap=False,
    )
    return replace(base, **overrides)


def _side_only_signature(amount=100.0):
    return TrafficSignature(
        "decode", amount, 0.0, 0, 0.0, n_side_requests=1
    )


def _pillar_parameters(**overrides):
    values = dict(
        side_hbm_path="pillar_bypass",
        side_hbm_owner_scope="adjacent",
        physical_adjacency=ADJACENCY_TP4,
        pillar_link_bw_bps=100.0,
        pillar_link_latency_s=0.0,
    )
    values.update(overrides)
    return _parameters(**values)


def test_pillar_remote_read_uses_owner_to_requester_direction_and_conserves_bytes():
    result = run_simulation(_side_only_signature(), _pillar_parameters())

    assert result.pillar_side_read_edge_bytes == 400.0
    assert result.side_hbm_read_h2h_edge_bytes == 0.0
    assert sum(
        metric.transferred_amount
        for name, metric in result.resource_metrics.items()
        if name.startswith("pillar_link")
    ) == 400.0
    assert all(
        result.resource_metrics[f"pillar_link[{owner},{requester}]"].transferred_amount
        == 100.0
        for owner, requester in ((1, 0), (0, 1), (3, 2), (2, 3))
    )


def test_adjacent_owner_selection_is_deterministic_round_robin():
    parameters = _pillar_parameters(
        physical_adjacency=((1, 2), (0, 3), (0, 3), (1, 2))
    )

    assert [
        _side_hbm_owner(0, request_index, parameters)
        for request_index in range(6)
    ] == [1, 2, 1, 2, 1, 2]


def test_pillar_resources_are_directed_and_opposite_directions_do_not_queue():
    env = simpy.Environment()
    resources = build_resource_layer(
        env=env,
        tp=2,
        r=0.0,
        local_hbm_stack_count=1,
        local_hbm_stack_bw_bps=100.0,
        local_hbm_stack_capacity_bytes=1_000.0,
        side_hbm_bw_bps=100.0,
        side_hbm_latency_s=0.0,
        h2h_bw_bps=100.0,
        h2h_latency_s=0.0,
        gpu_direct_bw_bps=100.0,
        gpu_direct_latency_s=0.0,
        gpu_compute_flops=100.0,
        physical_adjacency=((1,), (0,)),
        pillar_link_bw_bps=100.0,
        pillar_link_latency_s=0.0,
    )

    env.process(resources.pillar_link[(0, 1)].use(100.0))
    env.process(resources.pillar_link[(1, 0)].use(100.0))
    env.run()

    assert env.now == 1.0
    assert resources.pillar_link[(0, 1)] is not resources.pillar_link[(1, 0)]
    assert resources.pillar_link[(0, 1)].stats.total_queue_time_s == 0.0
    assert resources.pillar_link[(1, 0)].stats.total_queue_time_s == 0.0


@pytest.mark.parametrize(
    ("adjacency", "match"),
    [
        (None, "physical_adjacency"),
        (((1,), (0,), (3,)), "exactly one entry"),
        (((1,), (), (3,), (2,)), "at least one adjacent"),
        (((0,), (0,), (3,), (2,)), "self"),
        (((4,), (0,), (3,), (2,)), "range"),
        (((1,), (0,), (3,), (1,)), "symmetric"),
    ],
)
def test_adjacent_architecture_rejects_six_invalid_adjacency_shapes(adjacency, match):
    with pytest.raises(ValueError, match=match):
        run_simulation(
            _side_only_signature(),
            _parameters(
                side_hbm_path="pillar_bypass",
                side_hbm_owner_scope="adjacent",
                physical_adjacency=adjacency,
                pillar_link_bw_bps=100.0,
                pillar_link_latency_s=0.0,
            ),
        )


def test_ideal_pillar_matches_same_owner_independent_control():
    signature = _side_only_signature()
    pillar = run_simulation(
        signature,
        _pillar_parameters(pillar_link_bw_bps=1e30),
    )
    independent = run_simulation(
        signature,
        _parameters(
            side_hbm_path="independent",
            side_hbm_owner_scope="adjacent",
            physical_adjacency=ADJACENCY_TP4,
        ),
    )

    assert pillar.elapsed_s == pytest.approx(independent.elapsed_s, rel=1e-12)
    assert pillar.side_hbm_read_h2h_edge_bytes == 0.0
    assert independent.side_hbm_read_h2h_edge_bytes == 0.0


def test_tp1_pillar_is_noop_without_side_traffic_and_rejects_positive_side_traffic():
    no_side = TrafficSignature("decode", 0.0, 0.0, 0, 0.0)
    shared = run_simulation(no_side, _parameters(tp=1, r_remote=0.0))
    pillar = run_simulation(
        no_side,
        _parameters(
            tp=1,
            r_remote=0.0,
            side_hbm_path="pillar_bypass",
            side_hbm_owner_scope="adjacent",
        ),
    )
    assert pillar == shared

    with pytest.raises(ValueError, match="TP=1"):
        run_simulation(
            _side_only_signature(),
            _parameters(
                tp=1,
                side_hbm_path="pillar_bypass",
                side_hbm_owner_scope="adjacent",
            ),
        )


def test_tp4_pillar_is_a_performance_noop_when_side_traffic_is_zero():
    signature = TrafficSignature("decode", 100.0, 0.0, 0, 0.0)
    pillar = run_simulation(
        signature,
        _pillar_parameters(r_remote=0.0),
    )
    independent = run_simulation(
        signature,
        _parameters(r_remote=0.0, side_hbm_path="independent"),
    )

    assert pillar.elapsed_s == independent.elapsed_s
    assert pillar.ttft_s == independent.ttft_s
    assert pillar.tpot_s == independent.tpot_s
    assert pillar.pillar_side_read_edge_bytes == 0.0
    assert all(
        metric.transferred_amount == 0.0
        for name, metric in pillar.resource_metrics.items()
        if name.startswith("pillar_link")
    )


def test_crossing_energy_is_mutually_exclusive_while_side_datapath_is_common():
    config = load_config("configs/stage2_default.yaml")
    signature = _side_only_signature()
    common = dict(
        channels=config.channels,
        lane_rate_gbps=24.0,
        local_hbm_stack_count=1,
        elapsed_s=1.0,
        tp=4,
    )
    independent_result = run_simulation(
        signature, _parameters(side_hbm_path="independent")
    )
    shared_result = run_simulation(signature, _parameters())
    pillar_result = run_simulation(signature, _pillar_parameters())

    independent = calculate_channel_energy(
        independent_result.resource_metrics,
        side_hbm_path="independent",
        **common,
    )
    shared = calculate_channel_energy(
        shared_result.resource_metrics,
        side_hbm_path="shared_h2h",
        **common,
    )
    pillar = calculate_channel_energy(
        pillar_result.resource_metrics,
        side_hbm_path="pillar_bypass",
        pillar_link_energy_pj_per_bit=2.0,
        **common,
    )

    assert independent.side_hbm_energy_j == shared.side_hbm_energy_j
    assert shared.side_hbm_energy_j == pillar.side_hbm_energy_j
    assert independent.side_hbm_link_energy_j > 0.0
    assert independent.h2h_energy_j == independent.pillar_link_energy_j == 0.0
    assert shared.h2h_energy_j > 0.0
    assert shared.side_hbm_link_energy_j == shared.pillar_link_energy_j == 0.0
    assert pillar.pillar_link_energy_j == pytest.approx(400.0 * 8 * 2e-12)
    assert pillar.side_hbm_link_energy_j == pillar.h2h_energy_j == 0.0
    assert pillar.total_energy_j == pytest.approx(
        pillar.local_hbm_energy_j
        + pillar.local_hbm_idle_energy_j
        + pillar.side_hbm_energy_j
        + pillar.side_hbm_link_energy_j
        + pillar.h2h_energy_j
        + pillar.pillar_link_energy_j
        + pillar.gpu_direct_energy_j
    )


def test_existing_shared_and_independent_all_remote_results_are_unchanged():
    signature = _side_only_signature()

    shared = run_simulation(signature, _parameters())
    shared_explicit = run_simulation(
        signature,
        _parameters(side_hbm_owner_scope="all_remote"),
    )
    independent = run_simulation(
        signature, _parameters(side_hbm_path="independent")
    )
    independent_explicit = run_simulation(
        signature,
        _parameters(
            side_hbm_path="independent",
            side_hbm_owner_scope="all_remote",
        ),
    )

    assert shared == shared_explicit
    assert independent == independent_explicit


def test_d3_config_loads_exactly_three_paired_operating_points_and_adjacency():
    config = load_config("configs/stage2_d3_pillar_bypass.yaml")

    assert config.side_hbm_path_values == ("pillar_bypass",)
    assert config.side_hbm_owner_scope_values == ("adjacent",)
    assert config.channels.pillar_link is not None
    assert tuple(point.name for point in config.channels.pillar_link.operating_points) == (
        "conservative",
        "nominal",
        "aggressive",
    )
    assert set(config.physical_adjacency_by_tp) == {2, 4, 8}


def test_d3_config_runs_three_paired_points_without_cartesian_expansion():
    config = load_config("configs/stage2_d3_pillar_bypass.yaml")
    config = replace(
        config,
        workload=replace(config.workload, r_remote_values=(0.1,)),
    )

    rows = run_sweep(config)

    assert len(rows) == 3
    assert {row.pillar_operating_point for row in rows} == {
        "conservative",
        "nominal",
        "aggressive",
    }
    assert all(row.pillar_side_read_edge_bytes > 0.0 for row in rows)
    assert all(row.pillar_link_energy_j > 0.0 for row in rows)
    assert {row.pillar_link_energy_pj_per_bit for row in rows} == {1.0, 0.54, 0.30}
    assert all(not row.architecture_feasibility_applicable for row in rows)


def test_d3_config_rejects_selected_tp_without_explicit_adjacency(tmp_path):
    text = Path("configs/stage2_d3_pillar_bypass.yaml").read_text(
        encoding="utf-8"
    ).replace("  tp: 4\n", "  tp: 3\n")
    path = tmp_path / "missing_tp.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="not defined for TP=3"):
        load_config(path)


def test_pillar_parameters_are_required_only_when_pillar_path_is_selected(tmp_path):
    raw = yaml.safe_load(
        Path("configs/stage2_d3_pillar_bypass.yaml").read_text(encoding="utf-8")
    )
    raw["side_hbm_path"] = "shared_h2h"
    del raw["channels"]["pillar_link"]
    shared_path = tmp_path / "shared_adjacent.yaml"
    shared_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert load_config(shared_path).channels.pillar_link is None

    raw["side_hbm_path"] = "pillar_bypass"
    missing_pillar = tmp_path / "missing_pillar.yaml"
    missing_pillar.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="channels.pillar_link is required"):
        load_config(missing_pillar)


def test_recovery_fraction_contract():
    from analysis_d3_pillar_bypass import recovery_fraction

    assert recovery_fraction(10.0, 8.0, 6.0, side_hbm_bytes=1.0) == 0.5
    assert recovery_fraction(10.0, 10.0, 6.0, side_hbm_bytes=1.0) == 0.0
    assert recovery_fraction(10.0, 6.0, 6.0, side_hbm_bytes=1.0) == 1.0
    assert recovery_fraction(10.0, 5.0, 6.0, side_hbm_bytes=1.0) > 1.0
    assert recovery_fraction(10.0, 11.0, 6.0, side_hbm_bytes=1.0) < 0.0
    assert recovery_fraction(10.0, 8.0, 6.0, side_hbm_bytes=0.0) is None
    assert recovery_fraction(1.0, 1.0, math.nextafter(1.0, 2.0), 1.0) is None
