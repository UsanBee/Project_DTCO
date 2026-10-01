import csv
from dataclasses import replace

from test_layer_prefetch_simulator import _config


def test_cartesian_expansion_covers_every_declared_prefetch_axis():
    """Silently zipping independent sweep axes instead of product must fail."""
    from uhbm_simulator.prefetch_sweep import expand_prefetch_points

    config = _config()
    b0 = replace(
        config.architectures[0],
        name="B0",
        kind="b0_2_5d",
        side_hbm=None,
        side_link=None,
    )
    capacity_slow = replace(
        config.capacity_memory,
        name="slow",
        link=replace(config.capacity_memory.link, bw_bps=1e9),
    )
    capacity_fast = replace(
        config.capacity_memory,
        name="fast",
        link=replace(config.capacity_memory.link, bw_bps=2e9),
    )
    config = replace(
        config,
        architectures=(b0, config.architectures[0]),
        capacity_memory=(capacity_slow, capacity_fast),
        prefetch=replace(
            config.prefetch,
            weight_residency_policy_values=("streaming", "persistent_if_fit"),
            weight_chunk_bytes_values=(1024, 2048),
            kv_chunk_bytes_values=(256, 512),
            kv_writeback_chunk_bytes_values=(64, 128),
            max_inflight_chunks_per_lane_values=(1, None),
        ),
        workload=replace(
            config.workload,
            start_state_values=("cold_start", "steady_state"),
            batch_values=(1, 2),
            decode_tokens_values=(1, 2),
        ),
    )

    points = expand_prefetch_points(config)
    assert len(points) == 2**10
    identities = {
        (
            point.architectures[0].name,
            point.capacity_memory.name,
            point.workload.batch_values[0],
            point.workload.decode_tokens_values[0],
            point.workload.start_state_values[0],
            point.prefetch.weight_residency_policy_values[0],
            point.prefetch.weight_chunk_bytes_values[0],
            point.prefetch.kv_chunk_bytes_values[0],
            point.prefetch.kv_writeback_chunk_bytes_values[0],
            point.prefetch.max_inflight_chunks_per_lane_values[0],
        )
        for point in points
    }
    assert len(identities) == len(points)


def test_infeasible_rows_keep_identity_and_leave_simulation_metrics_none():
    """Dropping infeasible points or fabricating zero latency must fail."""
    from uhbm_simulator.prefetch_sweep import run_prefetch_sweep

    config = _config(kind="b0_2_5d")
    infeasible = replace(
        config.architectures[0], name="tiny", local_memory=replace(
            config.architectures[0].local_memory, capacity_bytes=1.0
        )
    )
    feasible = replace(config.architectures[0], name="large")
    rows = run_prefetch_sweep(
        replace(config, architectures=(infeasible, feasible))
    )
    by_name = {row.architecture: row for row in rows}

    assert set(by_name) == {"tiny", "large"}
    assert by_name["tiny"].capacity_feasible is False
    assert by_name["tiny"].elapsed_s is None
    assert by_name["tiny"].tpot_s is None
    assert by_name["tiny"].infeasibility_reason is not None
    assert by_name["large"].capacity_feasible is True
    assert by_name["large"].elapsed_s > 0.0


def test_summary_csv_contains_axes_and_metrics_but_no_detail_explosion(tmp_path):
    """Serializing token/layer/lane trace into each summary row must fail."""
    from uhbm_simulator.prefetch_sweep import (
        run_prefetch_sweep,
        write_prefetch_csv,
    )

    rows = run_prefetch_sweep(_config())
    output = tmp_path / "prefetch.csv"
    write_prefetch_csv(rows, output)
    with output.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))

    assert len(records) == 1
    record = records[0]
    assert record["architecture"] == "b2_uhbm"
    assert record["decode_tokens"] == "1"
    assert record["capacity_memory_profile"] == "scenario"
    assert "total_dynamic_energy_j" in record
    assert "path_metrics_json" in record
    assert all("token_timing" not in name for name in record)
    assert all("layer_timing" not in name for name in record)
    assert all("lane_ready" not in name for name in record)

