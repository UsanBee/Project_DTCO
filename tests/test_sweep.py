import csv
from dataclasses import replace
from pathlib import Path

import pytest

from uhbm_simulator.config import load_config
import uhbm_simulator.sweep as sweep_module
from uhbm_simulator.sweep import run_sweep, write_csv


SMALL_CONFIG = """
baseline: B2-hybrid
comm_mode: materialized
thermal_scenario: no_throttling
r: [0, 1]
channels:
  local_hbm:
    stack:
      bw_bps: 1.2e12
      capacity_bytes: 16e9
    count: [4]
    max_count: 4
    address_mapping:
      policy: uniform
      weights: null
  side_hbm:
    bw_bps: 4.8e12
    latency_s: 50e-9
  h2h:
    lane_rate_gbps: [48]
    latency_s: 20e-9
    n_pillar: [4, 32]
    n_pillar_max: 64
  gpu_direct:
    bw_bps: TBD
    latency_s: TBD
workload:
  phase: decode
  model: Llama-7B
  tp: 2
  pp: 1
  seq_len: 128
  batch: 1
  r_remote: [0]
"""


def test_sweep_expands_all_axes_and_exports_stats_csv(tmp_path):
    config_path = tmp_path / "small.yaml"
    config_path.write_text(SMALL_CONFIG, encoding="utf-8")
    rows = run_sweep(load_config(config_path))

    assert len(rows) == 4
    assert {row.r for row in rows} == {0.0, 1.0}
    assert {row.n_pillar for row in rows} == {4, 32}
    assert {row.local_hbm_count for row in rows} == {4}
    low_pillar_bw = next(row.h2h_bw_bps for row in rows if row.n_pillar == 4)
    high_pillar_bw = next(row.h2h_bw_bps for row in rows if row.n_pillar == 32)
    assert low_pillar_bw > high_pillar_bw
    assert all(row.tpot_s > 0 for row in rows)
    assert all(row.total_queue_time_s >= 0 for row in rows)
    r_zero = [row for row in rows if row.r == 0.0]
    r_one = [row for row in rows if row.r == 1.0]
    assert all(row.h2h_feasibility_checked for row in r_zero)
    assert all(row.required_h2h_bw_bps is not None for row in r_zero)
    assert all(row.architecture_feasible is not None for row in r_zero)
    assert all(not row.h2h_feasibility_checked for row in r_one)
    assert all(row.architecture_feasible is None for row in r_one)

    output = tmp_path / "results.csv"
    write_csv(rows, output)
    with output.open(newline="", encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))

    assert len(written) == 4
    assert "h2h_utilization" in written[0]
    assert "gpu_direct_utilization" in written[0]
    assert "local_hbm_count" in written[0]
    assert "local_hbm_utilization_max" in written[0]
    assert "architecture_feasible" in written[0]


def test_sweep_expands_local_hbm_count_axis(tmp_path):
    config_path = tmp_path / "count-sweep.yaml"
    config_path.write_text(
        SMALL_CONFIG.replace("count: [4]", "count: [1, 2, 4]"),
        encoding="utf-8",
    )

    rows = run_sweep(load_config(config_path))

    assert len(rows) == 12
    assert {row.local_hbm_count for row in rows} == {1, 2, 4}


def test_prefill_feasibility_uses_prefill_not_decode_signature(tmp_path):
    decode_path = tmp_path / "decode.yaml"
    prefill_path = tmp_path / "prefill.yaml"
    faster_local = SMALL_CONFIG.replace("bw_bps: 1.2e12", "bw_bps: 2.4e12", 1)
    decode_path.write_text(faster_local, encoding="utf-8")
    prefill_path.write_text(
        faster_local.replace("phase: decode", "phase: prefill"), encoding="utf-8"
    )

    decode_row = next(row for row in run_sweep(load_config(decode_path)) if row.r == 0)
    prefill_row = next(
        row for row in run_sweep(load_config(prefill_path)) if row.r == 0
    )

    assert prefill_row.required_h2h_bw_bps != decode_row.required_h2h_bw_bps


def test_hbm_count_sweep_example_runs_each_configured_count():
    config = load_config(Path("configs/stage2_hbm_count_sweep.yaml"))
    reduced = replace(
        config,
        r_values=(0.0,),
        channels=replace(
            config.channels,
            h2h=replace(
                config.channels.h2h,
                lane_rate_gbps=(config.channels.h2h.lane_rate_gbps[0],),
                n_pillar=(config.channels.h2h.n_pillar[0],),
            ),
        ),
        workload=replace(config.workload, r_remote_values=(0.0,)),
    )

    rows = run_sweep(reduced)

    assert len(rows) == 3
    assert {row.local_hbm_count for row in rows} == {1, 2, 4}


def test_breakeven_map_keeps_mixed_capacity_feasibility():
    config = load_config(Path("configs/stage2_breakeven_map.yaml"))
    reduced = replace(
        config,
        channels=replace(
            config.channels,
            h2h=replace(
                config.channels.h2h,
                lane_rate_gbps=(config.channels.h2h.lane_rate_gbps[0],),
                n_pillar=(config.channels.h2h.n_pillar[0],),
            ),
        ),
    )

    rows = run_sweep(reduced)

    assert config.workload.model == "Llama-70B"
    assert config.channels.local_hbm.count_values == (1, 2, 4)
    assert any(not row.capacity_feasible for row in rows)
    assert any(row.capacity_feasible for row in rows)


def _capacity_mixed_config(tmp_path):
    config_path = tmp_path / "capacity-mixed.yaml"
    config_path.write_text(
        SMALL_CONFIG.replace("r: [0, 1]", "r: [0]")
        .replace("capacity_bytes: 16e9", "capacity_bytes: 2e9")
        .replace("count: [4]", "count: [1, 4]")
        .replace("n_pillar: [4, 32]", "n_pillar: [4]"),
        encoding="utf-8",
    )
    return load_config(config_path)


def test_capacity_overflow_is_recorded_without_aborting_sweep(tmp_path):
    rows = run_sweep(_capacity_mixed_config(tmp_path))

    infeasible = next(row for row in rows if row.local_hbm_count == 1)
    assert infeasible.capacity_feasible is False
    assert infeasible.h2h_feasibility_checked is True
    assert infeasible.required_h2h_bw_bps is not None
    assert infeasible.architecture_feasible is not None
    assert infeasible.ttft_s is None
    assert infeasible.tpot_s is None
    assert infeasible.communication_time_s is None
    assert infeasible.overlap_s is None
    assert infeasible.local_hbm_utilization is None
    assert infeasible.local_hbm_utilization_max is None
    assert infeasible.side_hbm_utilization is None
    assert infeasible.h2h_utilization is None
    assert infeasible.gpu_direct_utilization is None
    assert infeasible.gpu_compute_utilization is None
    assert infeasible.total_queue_time_s is None

    output = tmp_path / "capacity.csv"
    write_csv(rows, output)
    with output.open(newline="", encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))
    infeasible_csv = next(row for row in written if row["local_hbm_count"] == "1")
    assert infeasible_csv["capacity_feasible"] == "False"
    assert infeasible_csv["ttft_s"] == ""


def test_capacity_feasible_point_is_simulated_normally(tmp_path):
    rows = run_sweep(_capacity_mixed_config(tmp_path))

    feasible = next(row for row in rows if row.local_hbm_count == 4)
    assert feasible.capacity_feasible is True
    assert feasible.ttft_s is not None
    assert feasible.tpot_s is not None
    assert feasible.local_hbm_utilization is not None


def test_sweep_does_not_hide_noncapacity_value_errors(tmp_path, monkeypatch):
    def raise_unrelated_error(*args, **kwargs):
        raise ValueError("unrelated simulator bug")

    monkeypatch.setattr(sweep_module, "run_simulation", raise_unrelated_error)

    with pytest.raises(ValueError, match="unrelated simulator bug"):
        run_sweep(_capacity_mixed_config(tmp_path))
