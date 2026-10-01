"""D-3 pillar-bypass analysis over the established decode eligibility grid.

The script reuses the capacity-fullgrid workload/capacity definitions. The
eligibility rule is exactly: decode workloads for which B0 is feasible, crossed
with the first 12 capacity multipliers (0.02 through 0.50). With the corrected
D-1/D-2 model this is 39 workloads x 12 capacities = 468 points; the expanded
pillar operating-point simulations are reported separately and are not called
"468 points".

Pillar BW/latency/energy values are Stage 3 HFSS/ADS pre-calibration
placeholders. No N_signal-to-BW or N_power/PDN model is inferred here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import statistics

from analysis_capacity_breakeven_fullgrid import (
    CAPACITY_MULTIPLIERS,
    MODEL_NAME,
    all_workloads,
    b0_params,
    b2_params,
)
from uhbm_simulator.config import PillarLinkOperatingPointConfig, load_config
from uhbm_simulator.simulator import (
    LocalHBMCapacityError,
    SideHBMCapacityError,
    SimulationParameters,
    calculate_channel_energy,
    run_simulation,
)


CONFIG = Path("configs/stage2_d3_pillar_bypass.yaml")
DETAIL_CSV = Path("results/d3_pillar_bypass_checkpoint_rows.csv")
PRODUCTION_CSV = Path("results/d3_pillar_bypass_production.csv")
CONTROLLED_CSV = Path("results/d3_pillar_bypass_controlled_adjacent.csv")
TP2_CSV = Path("results/d3_pillar_bypass_tp2_contention_isolation.csv")
SUMMARY_CSV = Path("results/d3_pillar_bypass_summary.csv")
CHECKPOINT = Path("results/d3_pillar_bypass_checkpoint.json")


def recovery_fraction(
    shared_time_s: float,
    pillar_time_s: float,
    independent_time_s: float,
    side_hbm_bytes: float,
) -> float | None:
    """Return recovered shared-path penalty, or None when undefined/noisy."""
    if side_hbm_bytes <= 0.0:
        return None
    denominator = shared_time_s - independent_time_s
    tolerance = max(
        1e-15,
        1e-12 * max(abs(shared_time_s), abs(independent_time_s)),
    )
    if abs(denominator) <= tolerance:
        return None
    return (shared_time_s - pillar_time_s) / denominator


@dataclass(frozen=True)
class D3DetailRow:
    tp: int
    seq_len: int
    batch: int
    capacity_multiplier: float
    capacity_gb: float
    pillar_operating_point: str
    pillar_link_bw_bps: float
    pillar_link_latency_s: float
    pillar_link_energy_pj_per_bit: float
    b0_tpot_s: float
    side_hbm_bytes_per_rank: float
    production_shared_all_remote_tpot_s: float
    production_pillar_adjacent_tpot_s: float
    production_shared_wins_vs_b0: bool
    production_pillar_wins_vs_b0: bool
    controlled_shared_adjacent_tpot_s: float | None
    controlled_independent_adjacent_tpot_s: float | None
    controlled_pillar_adjacent_tpot_s: float | None
    controlled_recovery_fraction: float | None
    pillar_side_read_edge_bytes_system: float
    pillar_link_utilization_mean: float
    pillar_link_utilization_max: float
    side_hbm_datapath_energy_j: float
    pillar_link_energy_j: float
    pillar_total_energy_j: float
    tp2_isolation: bool


def _run(
    signature,
    tp: int,
    capacity_bytes: float,
    *,
    path: str,
    owner_scope: str,
    adjacency: tuple[tuple[int, ...], ...] | None,
    point: PillarLinkOperatingPointConfig | None = None,
):
    parameters: SimulationParameters = replace(
        b2_params(tp, capacity_bytes, "full_overlap"),
        side_hbm_path=path,
        side_hbm_owner_scope=owner_scope,
        physical_adjacency=adjacency,
        pillar_link_bw_bps=(point.bw_bps if point is not None else None),
        pillar_link_latency_s=(point.latency_s if point is not None else None),
    )
    try:
        return run_simulation(signature, parameters)
    except (LocalHBMCapacityError, SideHBMCapacityError) as error:
        raise RuntimeError(
            "D-3 capacity-overflow point unexpectedly became infeasible"
        ) from error


def _side_bytes_per_rank(result, tp: int) -> float:
    return math.fsum(
        metric.transferred_amount
        for name, metric in result.resource_metrics.items()
        if name.startswith("side_hbm")
    ) / tp


def _utilization(result, prefix: str) -> tuple[float, float]:
    values = [
        metric.utilization
        for name, metric in result.resource_metrics.items()
        if name.startswith(prefix)
    ]
    return (
        (math.fsum(values) / len(values) if values else 0.0),
        max(values, default=0.0),
    )


def eligible_workloads():
    """Yield the original 468-rule workload set (before capacity expansion)."""
    for phase, builder, tp, seq_len, batch in all_workloads():
        if phase != "decode":
            continue
        signature = builder(MODEL_NAME, tp, seq_len, batch)
        try:
            b0_result = run_simulation(signature, b0_params(tp))
        except (LocalHBMCapacityError, SideHBMCapacityError):
            continue
        yield signature, tp, seq_len, batch, b0_result.tpot_s


def process_workload(signature, tp, seq_len, batch, b0_tpot_s, config):
    pillar_config = config.channels.pillar_link
    if pillar_config is None:
        raise ValueError("D-3 config must define channels.pillar_link")
    adjacency = config.physical_adjacency_by_tp[tp]
    footprint = signature.local_footprint_bytes
    if footprint is None:
        raise ValueError("D-3 requires an explicit resident footprint")

    rows = []
    for multiplier in CAPACITY_MULTIPLIERS[:12]:
        capacity = footprint * multiplier
        shared_all_remote = _run(
            signature,
            tp,
            capacity,
            path="shared_h2h",
            owner_scope="all_remote",
            adjacency=None,
        )
        shared_adjacent = _run(
            signature,
            tp,
            capacity,
            path="shared_h2h",
            owner_scope="adjacent",
            adjacency=adjacency,
        )
        independent_adjacent = _run(
            signature,
            tp,
            capacity,
            path="independent",
            owner_scope="adjacent",
            adjacency=adjacency,
        )
        side_bytes = _side_bytes_per_rank(shared_adjacent, tp)
        for point in pillar_config.operating_points:
            pillar = _run(
                signature,
                tp,
                capacity,
                path="pillar_bypass",
                owner_scope="adjacent",
                adjacency=adjacency,
                point=point,
            )
            pillar_energy = calculate_channel_energy(
                pillar.resource_metrics,
                config.channels,
                lane_rate_gbps=config.channels.h2h.lane_rate_gbps[0],
                local_hbm_stack_count=config.channels.local_hbm.count_values[0],
                elapsed_s=pillar.elapsed_s,
                tp=tp,
                side_hbm_path="pillar_bypass",
                pillar_link_energy_pj_per_bit=point.energy_pj_per_bit,
            )
            pillar_util_mean, pillar_util_max = _utilization(
                pillar, "pillar_link"
            )
            controlled = recovery_fraction(
                shared_adjacent.tpot_s,
                pillar.tpot_s,
                independent_adjacent.tpot_s,
                side_bytes,
            )
            rows.append(
                D3DetailRow(
                    tp=tp,
                    seq_len=seq_len,
                    batch=batch,
                    capacity_multiplier=multiplier,
                    capacity_gb=capacity / 1e9,
                    pillar_operating_point=point.name,
                    pillar_link_bw_bps=point.bw_bps,
                    pillar_link_latency_s=point.latency_s,
                    pillar_link_energy_pj_per_bit=point.energy_pj_per_bit,
                    b0_tpot_s=b0_tpot_s,
                    side_hbm_bytes_per_rank=side_bytes,
                    production_shared_all_remote_tpot_s=shared_all_remote.tpot_s,
                    production_pillar_adjacent_tpot_s=pillar.tpot_s,
                    production_shared_wins_vs_b0=(
                        shared_all_remote.tpot_s <= b0_tpot_s
                    ),
                    production_pillar_wins_vs_b0=pillar.tpot_s <= b0_tpot_s,
                    controlled_shared_adjacent_tpot_s=shared_adjacent.tpot_s,
                    controlled_independent_adjacent_tpot_s=(
                        independent_adjacent.tpot_s
                    ),
                    controlled_pillar_adjacent_tpot_s=pillar.tpot_s,
                    controlled_recovery_fraction=controlled,
                    pillar_side_read_edge_bytes_system=(
                        pillar.pillar_side_read_edge_bytes
                    ),
                    pillar_link_utilization_mean=pillar_util_mean,
                    pillar_link_utilization_max=pillar_util_max,
                    side_hbm_datapath_energy_j=(
                        pillar_energy.side_hbm_energy_j
                    ),
                    pillar_link_energy_j=pillar_energy.pillar_link_energy_j,
                    pillar_total_energy_j=pillar_energy.total_energy_j,
                    tp2_isolation=tp == 2,
                )
            )
    return rows


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0], ordered[0], ordered[0]
    q1, median, q3 = statistics.quantiles(
        ordered, n=4, method="inclusive"
    )[0], statistics.median(ordered), statistics.quantiles(
        ordered, n=4, method="inclusive"
    )[2]
    return q1, median, q3


def write_summary(rows: list[D3DetailRow], path: Path = SUMMARY_CSV) -> None:
    groups = {}
    for row in rows:
        groups.setdefault((row.tp, row.pillar_operating_point), []).append(row)
        groups.setdefault(("all", row.pillar_operating_point), []).append(row)
    summary = []
    for (tp, point), grouped in sorted(groups.items(), key=lambda item: str(item[0])):
        recoveries = [
            row.controlled_recovery_fraction
            for row in grouped
            if row.controlled_recovery_fraction is not None
        ]
        q1, median, q3 = (
            _quartiles(recoveries) if recoveries else (None, None, None)
        )
        summary.append(
            {
                "tp": tp,
                "comparison": (
                    "all_eligible"
                    if tp == "all"
                    else (
                        "tp2_contention_isolation"
                        if tp == 2
                        else "controlled_adjacent"
                    )
                ),
                "pillar_operating_point": point,
                "n_eligibility_points": len(grouped),
                "production_shared_wins_vs_b0": sum(
                    row.production_shared_wins_vs_b0 for row in grouped
                ),
                "production_pillar_wins_vs_b0": sum(
                    row.production_pillar_wins_vs_b0 for row in grouped
                ),
                "recovery_fraction_median": median,
                "recovery_fraction_q1": q1,
                "recovery_fraction_q3": q3,
                "recovery_fraction_none_count": len(grouped) - len(recoveries),
                "recovery_fraction_none_fraction": (
                    (len(grouped) - len(recoveries)) / len(grouped)
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def _write_view(
    rows: list[D3DetailRow],
    path: Path,
    fields: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            writer.writerow({field: values[field] for field in fields})


def write_separate_reports(rows: list[D3DetailRow]) -> None:
    identifiers = (
        "tp",
        "seq_len",
        "batch",
        "capacity_multiplier",
        "capacity_gb",
        "pillar_operating_point",
        "pillar_link_bw_bps",
        "pillar_link_latency_s",
        "pillar_link_energy_pj_per_bit",
        "b0_tpot_s",
        "side_hbm_bytes_per_rank",
    )
    pillar_stats = (
        "pillar_side_read_edge_bytes_system",
        "pillar_link_utilization_mean",
        "pillar_link_utilization_max",
        "side_hbm_datapath_energy_j",
        "pillar_link_energy_j",
        "pillar_total_energy_j",
    )
    _write_view(
        rows,
        PRODUCTION_CSV,
        identifiers
        + (
            "production_shared_all_remote_tpot_s",
            "production_pillar_adjacent_tpot_s",
            "production_shared_wins_vs_b0",
            "production_pillar_wins_vs_b0",
        )
        + pillar_stats,
    )
    _write_view(
        [row for row in rows if row.tp in {4, 8}],
        CONTROLLED_CSV,
        identifiers
        + (
            "controlled_shared_adjacent_tpot_s",
            "controlled_independent_adjacent_tpot_s",
            "controlled_pillar_adjacent_tpot_s",
            "controlled_recovery_fraction",
        )
        + pillar_stats,
    )
    _write_view(
        [row for row in rows if row.tp == 2],
        TP2_CSV,
        identifiers
        + (
            "production_shared_all_remote_tpot_s",
            "controlled_independent_adjacent_tpot_s",
            "production_pillar_adjacent_tpot_s",
            "controlled_recovery_fraction",
        )
        + pillar_stats,
    )


def _read_detail(path: Path) -> list[D3DetailRow]:
    rows = []
    if not path.exists():
        return rows
    optional_float = {
        "controlled_shared_adjacent_tpot_s",
        "controlled_independent_adjacent_tpot_s",
        "controlled_pillar_adjacent_tpot_s",
        "controlled_recovery_fraction",
    }
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            converted = {}
            for field in D3DetailRow.__dataclass_fields__:
                value = raw[field]
                if field in {"tp", "seq_len", "batch"}:
                    converted[field] = int(value)
                elif field in {
                    "production_shared_wins_vs_b0",
                    "production_pillar_wins_vs_b0",
                    "tp2_isolation",
                }:
                    converted[field] = value == "True"
                elif field == "pillar_operating_point":
                    converted[field] = value
                elif field in optional_float and value == "":
                    converted[field] = None
                else:
                    converted[field] = float(value)
            rows.append(D3DetailRow(**converted))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        for path in (
            DETAIL_CSV,
            PRODUCTION_CSV,
            CONTROLLED_CSV,
            TP2_CSV,
            SUMMARY_CSV,
            CHECKPOINT,
        ):
            if path.exists():
                path.unlink()

    config = load_config(CONFIG)
    workloads = list(eligible_workloads())
    state = {"next_index": 0}
    if CHECKPOINT.exists():
        state = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    start = int(state["next_index"])
    end = len(workloads) if args.chunk is None else min(
        len(workloads), start + args.chunk
    )
    DETAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not DETAIL_CSV.exists()
    with DETAIL_CSV.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(D3DetailRow.__dataclass_fields__)
        )
        if is_new:
            writer.writeheader()
        for index in range(start, end):
            rows = process_workload(*workloads[index], config)
            writer.writerows(asdict(row) for row in rows)
            stream.flush()
            CHECKPOINT.write_text(
                json.dumps({"next_index": index + 1}), encoding="utf-8"
            )
            _, tp, seq_len, batch, _ = workloads[index]
            print(
                f"progress {index + 1}/{len(workloads)}: "
                f"tp={tp} seq_len={seq_len} batch={batch}",
                flush=True,
            )
    all_rows = _read_detail(DETAIL_CSV)
    if end == len(workloads):
        write_separate_reports(all_rows)
        write_summary(all_rows)
        print(
            f"complete: {len(workloads)} workloads, "
            f"{len(all_rows)} expanded detail rows",
            flush=True,
        )


if __name__ == "__main__":
    main()
