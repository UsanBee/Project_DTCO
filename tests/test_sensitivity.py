import importlib.util
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from uhbm_simulator.config import load_config
from uhbm_simulator.sweep import run_sweep
import uhbm_simulator.sensitivity as sensitivity


def test_sensitivity_analysis_module_is_available():
    assert importlib.util.find_spec("uhbm_simulator.sensitivity") is not None


def test_baseline_config_reproduces_the_approved_feasible_point():
    assert hasattr(sensitivity, "build_baseline_config")
    config = load_config(Path("configs/stage2_default.yaml"))

    baseline_config = sensitivity.build_baseline_config(config)
    rows = run_sweep(baseline_config)

    assert len(rows) == 1
    row = rows[0]
    assert row.r == 0.5
    assert row.r_remote == 0.1
    assert row.comm_mode == "materialized"
    assert row.thermal_scenario == "no_throttling"
    assert row.local_hbm_count == 4
    assert baseline_config.channels.local_hbm.address_mapping.policy == "uniform"
    assert row.lane_rate_gbps == 24.0
    assert row.n_pillar == 8
    assert row.batch == 8
    assert (
        baseline_config.channels.side_hbm.link_energy_pj_per_bit
        == config.channels.side_hbm.link_energy_pj_per_bit
        == 10.0
    )
    assert (
        baseline_config.channels.local_hbm.stack.hbm_idle_power_w
        == config.channels.local_hbm.stack.hbm_idle_power_w
        == 0.4
    )
    assert row.capacity_feasible is True
    assert row.tpot_s is not None


def test_relative_range_ignores_infeasible_points_and_uses_baseline():
    assert hasattr(sensitivity, "metric_range")

    minimum, maximum, relative = sensitivity.metric_range(
        (1.0, None, 3.0), baseline=2.0
    )

    assert minimum == 1.0
    assert maximum == 3.0
    assert relative == 1.0


def test_metric_range_returns_none_when_prefill_decode_metric_is_all_none():
    minimum, maximum, relative = sensitivity.metric_range(
        (None, None), baseline=None
    )

    assert minimum is None
    assert maximum is None
    assert relative is None


def test_oat_report_covers_every_axis_and_preserves_semantics(tmp_path):
    assert hasattr(sensitivity, "run_sensitivity")
    assert hasattr(sensitivity, "write_sensitivity_csv")
    config = load_config(Path("configs/stage2_default.yaml"))

    rows = sensitivity.run_sensitivity(config)

    by_axis = {row.axis: row for row in rows}
    expected_point_counts = {
        "r": 5,
        "r_remote": 4,
        "comm_mode": 2,
        "thermal_scenario": 3,
        "local_hbm_count": 4,
        "address_mapping.policy": 2,
        "h2h.lane_rate_gbps": 3,
        "h2h.n_pillar": 4,
        "batch": 5,
        "side_hbm.link_energy_pj_per_bit": 4,
        "local_hbm.stack.hbm_idle_power_w": 3,
        "side_hbm_path": 1,
    }
    assert set(by_axis) == set(expected_point_counts)
    for axis, expected_count in expected_point_counts.items():
        row = by_axis[axis]
        assert row.n_swept_points == expected_count
        assert len(json.loads(row.swept_values)) == expected_count
        assert row.capacity_infeasible_fraction == (
            row.n_capacity_infeasible_points / expected_count
        )

    structurally_unmodeled = {
        "r",
        "r_remote",
        "comm_mode",
        "thermal_scenario",
        "address_mapping.policy",
        "side_hbm.link_energy_pj_per_bit",
        "local_hbm.stack.hbm_idle_power_w",
        "side_hbm_path",
    }
    for axis in structurally_unmodeled:
        assert by_axis[axis].architecture_feasibility_applicable is False
        assert by_axis[axis].flips_architecture_feasible is False
    for axis in set(expected_point_counts) - structurally_unmodeled:
        assert by_axis[axis].architecture_feasibility_applicable is True

    assert all(
        row.architecture_feasibility_semantics
        == "native_r0_h2h_feasibility_independent_of_actual_r"
        for row in rows
    )
    mapping_notes = by_axis["address_mapping.policy"].notes.lower()
    assert "intentional worst-case hotspot" in mapping_notes
    assert "0.7, 0.1, 0.1, 0.1" in mapping_notes

    side_link = by_axis["side_hbm.link_energy_pj_per_bit"]
    assert side_link.baseline_value == "10.0"
    assert json.loads(side_link.swept_values) == [0.54, 10.0, 20.0, 40.0]
    assert side_link.tpot_s_relative_range == 0.0
    assert side_link.communication_time_s_relative_range == 0.0
    # The production baseline uses shared_h2h, where side-link energy is zero
    # and traversed edge bytes are charged to h2h_energy_j instead.
    assert side_link.energy_per_token_j_relative_range == 0.0

    local_idle = by_axis["local_hbm.stack.hbm_idle_power_w"]
    assert local_idle.baseline_value == "0.4"
    assert json.loads(local_idle.swept_values) == [0.4, 2.0, 5.2]
    assert local_idle.tpot_s_relative_range == 0.0
    assert local_idle.communication_time_s_relative_range == 0.0
    assert local_idle.energy_per_token_j_relative_range > 0.0

    output = tmp_path / "sensitivity.csv"
    sensitivity.write_sensitivity_csv(rows, output)
    with output.open(newline="", encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))
    assert len(written) == 12
    assert "architecture_feasibility_applicable" in written[0]
    assert "capacity_infeasible_fraction" in written[0]
    assert "energy_per_token_j_min" in written[0]
    assert "energy_per_token_j_max" in written[0]
    assert "energy_per_token_j_relative_range" in written[0]
    assert "decode_throughput_tokens_per_s_min" in written[0]
    assert "decode_throughput_tokens_per_s_max" in written[0]
    assert "decode_throughput_tokens_per_s_relative_range" in written[0]


def test_module_cli_writes_one_row_per_axis(tmp_path):
    config_text = Path("configs/stage2_default.yaml").read_text(encoding="utf-8")
    config_text = (
        config_text.replace("r: [0, 0.25, 0.5, 0.75, 1.0]", "r: [0.5]")
        .replace("lane_rate_gbps: [12, 24, 32]", "lane_rate_gbps: [24]")
        .replace("n_pillar: [4, 8, 16, 32]", "n_pillar: [8]")
        .replace("model: Llama-70B", "model: Llama-7B")
        .replace("r_remote: [0, 0.1, 0.25, 0.5]", "r_remote: [0.1]")
        .replace("batch: [1, 4, 8, 16, 32]", "batch: [1]")
    )
    config_path = tmp_path / "sensitivity-small.yaml"
    output_path = tmp_path / "sensitivity.csv"
    config_path.write_text(config_text, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "uhbm_simulator.sensitivity",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Analyzed 12 sensitivity axes" in completed.stdout
    with output_path.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 12


def test_oat_calls_isolate_axes_and_use_native_r_zero_shadows(monkeypatch):
    config = load_config(Path("configs/stage2_default.yaml"))
    baseline = sensitivity.build_baseline_config(config)
    template = run_sweep(baseline)[0]
    calls = []

    def state(point_config):
        mapping = point_config.channels.local_hbm.address_mapping
        return {
            "r": point_config.r_values[0],
            "r_remote": point_config.workload.r_remote_values[0],
            "comm_mode": point_config.comm_mode,
            "thermal_scenario": point_config.thermal_scenario,
            "local_hbm_count": point_config.channels.local_hbm.count_values[0],
            "address_mapping.policy": (mapping.policy, mapping.weights),
            "h2h.lane_rate_gbps": point_config.channels.h2h.lane_rate_gbps[0],
            "h2h.n_pillar": point_config.channels.h2h.n_pillar[0],
            "batch": point_config.workload.batch_values[0],
            "side_hbm.link_energy_pj_per_bit": (
                point_config.channels.side_hbm.link_energy_pj_per_bit
            ),
            "local_hbm.stack.hbm_idle_power_w": (
                point_config.channels.local_hbm.stack.hbm_idle_power_w
            ),
            "side_hbm_path": point_config.side_hbm_path_values[0],
        }

    def controlled_sweep(point_config):
        calls.append(point_config)
        point = state(point_config)
        capacity_feasible = (
            point["local_hbm_count"] != 1
            and point["address_mapping.policy"][0] != "weighted"
        )
        architecture_feasible = (
            point["h2h.lane_rate_gbps"] >= 24.0
            if point["r"] == 0.0
            else None
        )
        return [
            replace(
                template,
                r=point["r"],
                r_remote=point["r_remote"],
                comm_mode=point["comm_mode"],
                thermal_scenario=point["thermal_scenario"],
                local_hbm_count=point["local_hbm_count"],
                lane_rate_gbps=point["h2h.lane_rate_gbps"],
                n_pillar=point["h2h.n_pillar"],
                batch=point["batch"],
                side_hbm_path=point["side_hbm_path"],
                architecture_feasibility_applicable=(
                    point["side_hbm_path"] == "independent"
                ),
                capacity_feasible=capacity_feasible,
                architecture_feasible=architecture_feasible,
                tpot_s=(
                    float(point["local_hbm_count"])
                    if capacity_feasible
                    else None
                ),
                communication_time_s=(
                    float(point["h2h.lane_rate_gbps"])
                    if capacity_feasible
                    else None
                ),
                energy_per_token_j=(
                    10.0 * float(point["local_hbm_count"])
                    if capacity_feasible
                    else None
                ),
                decode_throughput_tokens_per_s=(
                    100.0 * float(point["local_hbm_count"])
                    if capacity_feasible
                    else None
                ),
            )
        ]

    monkeypatch.setattr(sensitivity, "run_sweep", controlled_sweep)

    rows = sensitivity.run_sensitivity(config)

    by_axis = {row.axis: row for row in rows}
    assert by_axis["local_hbm_count"].flips_capacity_feasible is True
    assert by_axis["local_hbm_count"].n_capacity_infeasible_points == 1
    assert by_axis["local_hbm_count"].capacity_infeasible_fraction == 0.25
    assert by_axis["local_hbm_count"].tpot_s_min == 2.0
    assert by_axis["local_hbm_count"].tpot_s_max == 4.0
    assert by_axis["local_hbm_count"].tpot_s_relative_range == 0.5
    assert by_axis["local_hbm_count"].energy_per_token_j_min == 20.0
    assert by_axis["local_hbm_count"].energy_per_token_j_max == 40.0
    assert by_axis["local_hbm_count"].energy_per_token_j_relative_range == 0.5
    assert (
        by_axis["local_hbm_count"].decode_throughput_tokens_per_s_min
        == 200.0
    )
    assert (
        by_axis["local_hbm_count"].decode_throughput_tokens_per_s_max
        == 400.0
    )
    assert (
        by_axis[
            "local_hbm_count"
        ].decode_throughput_tokens_per_s_relative_range
        == 0.5
    )
    assert by_axis["address_mapping.policy"].flips_capacity_feasible is True
    assert by_axis["address_mapping.policy"].n_capacity_infeasible_points == 1
    assert by_axis["h2h.lane_rate_gbps"].flips_architecture_feasible is True
    assert by_axis["r"].flips_architecture_feasible is False

    baseline_state = state(baseline)
    assert state(calls[0]) == baseline_state
    assert state(calls[1]) == baseline_state
    cursor = 2
    for row in rows:
        values = json.loads(row.swept_values)
        for value in values:
            real_state = state(calls[cursor])
            shadow_state = state(calls[cursor + 1])
            expected_real = dict(baseline_state)
            expected_real[row.axis] = (
                (
                    value,
                    sensitivity.HOTSPOT_WEIGHTS if value == "weighted" else None,
                )
                if row.axis == "address_mapping.policy"
                else value
            )
            assert real_state == expected_real
            expected_shadow = dict(real_state)
            expected_shadow["r"] = 0.0
            assert shadow_state == expected_shadow
            cursor += 2
    assert cursor == len(calls)
