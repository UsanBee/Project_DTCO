"""One-at-a-time sensitivity analysis for Stage 2 sweep axes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Iterable, Sequence

from uhbm_simulator.config import (
    ARUNKUMAR2019_ON_BOARD_LINK_ENERGY_PJ_PER_BIT,
    ARUNKUMAR2019_ON_PACKAGE_LINK_ENERGY_PJ_PER_BIT,
    CROSS_METHODOLOGY_HIGH_REFERENCE_IDLE_POWER_W,
    DEFAULT_HBM_IDLE_POWER_W_PER_STACK,
    AddressMappingConfig,
    SimulationConfig,
    load_config,
)
from uhbm_simulator.sweep import SweepRow, run_sweep


AXES = (
    "r",
    "r_remote",
    "comm_mode",
    "thermal_scenario",
    "local_hbm_count",
    "address_mapping.policy",
    "h2h.lane_rate_gbps",
    "h2h.n_pillar",
    "batch",
    "side_hbm.link_energy_pj_per_bit",
    "local_hbm.stack.hbm_idle_power_w",
    "side_hbm_path",
)
COMM_MODE_VALUES = ("materialized", "cut_through")
THERMAL_SCENARIO_VALUES = (
    "no_throttling",
    "mild_throttling",
    "severe_throttling",
)
ADDRESS_MAPPING_VALUES = ("uniform", "weighted")
HOTSPOT_WEIGHTS = (0.7, 0.1, 0.1, 0.1)
SIDE_HBM_LINK_ENERGY_VALUES = (
    ARUNKUMAR2019_ON_PACKAGE_LINK_ENERGY_PJ_PER_BIT,
    ARUNKUMAR2019_ON_BOARD_LINK_ENERGY_PJ_PER_BIT,
    20.0,
    40.0,
)
LOCAL_HBM_IDLE_POWER_VALUES = (
    DEFAULT_HBM_IDLE_POWER_W_PER_STACK,
    2.0,
    # Normalize the cross-paper expression for stable human-facing CSV text.
    round(CROSS_METHODOLOGY_HIGH_REFERENCE_IDLE_POWER_W, 1),
)
ARCHITECTURE_FEASIBILITY_SEMANTICS = (
    "native_r0_h2h_feasibility_independent_of_actual_r"
)
ARCHITECTURE_APPLICABLE_AXES = {
    "local_hbm_count",
    "h2h.lane_rate_gbps",
    "h2h.n_pillar",
    "batch",
}


@dataclass(frozen=True)
class SensitivityRow:
    axis: str
    baseline_value: str
    swept_values: str
    n_swept_points: int
    tpot_s_min: float | None
    tpot_s_max: float | None
    tpot_s_relative_range: float | None
    communication_time_s_min: float | None
    communication_time_s_max: float | None
    communication_time_s_relative_range: float | None
    energy_per_token_j_min: float | None
    energy_per_token_j_max: float | None
    energy_per_token_j_relative_range: float | None
    decode_throughput_tokens_per_s_min: float | None
    decode_throughput_tokens_per_s_max: float | None
    decode_throughput_tokens_per_s_relative_range: float | None
    flips_capacity_feasible: bool
    flips_architecture_feasible: bool
    n_capacity_infeasible_points: int
    capacity_infeasible_fraction: float
    architecture_feasibility_applicable: bool
    architecture_feasibility_semantics: str
    notes: str


def _nearest(values: Iterable[float | int], target: float):
    values = tuple(values)
    if not values:
        raise ValueError("sensitivity axis cannot be empty")
    return min(values, key=lambda value: abs(float(value) - target))


def build_baseline_config(config: SimulationConfig) -> SimulationConfig:
    """Freeze the approved central Stage 2 sensitivity baseline."""
    local_hbm = replace(
        config.channels.local_hbm,
        stack=replace(
            config.channels.local_hbm.stack,
            hbm_idle_power_w=DEFAULT_HBM_IDLE_POWER_W_PER_STACK,
        ),
        count_values=(min(4, config.channels.local_hbm.max_count),),
        address_mapping=AddressMappingConfig(policy="uniform", weights=None),
    )
    h2h = replace(
        config.channels.h2h,
        lane_rate_gbps=(
            float(_nearest(config.channels.h2h.lane_rate_gbps, 24.0)),
        ),
        n_pillar=(int(_nearest(config.channels.h2h.n_pillar, 8.0)),),
    )
    frozen = replace(
        config,
        comm_mode="materialized",
        thermal_scenario="no_throttling",
        side_hbm_path_values=("shared_h2h",),
        r_values=(float(_nearest(config.r_values, 0.5)),),
        channels=replace(
            config.channels,
            local_hbm=local_hbm,
            side_hbm=replace(
                config.channels.side_hbm,
                link_energy_pj_per_bit=(
                    ARUNKUMAR2019_ON_BOARD_LINK_ENERGY_PJ_PER_BIT
                ),
            ),
            h2h=h2h,
        ),
        workload=replace(
            config.workload,
            r_remote_values=(
                float(_nearest(config.workload.r_remote_values, 0.1)),
            ),
            batch_values=(config.workload.batch_values[len(config.workload.batch_values) // 2],),
        ),
    )
    midpoint = len(config.workload.batch_values) // 2
    candidates = sorted(
        enumerate(config.workload.batch_values),
        key=lambda item: (abs(item[0] - midpoint), item[0]),
    )
    for _, batch in candidates:
        candidate = replace(
            frozen,
            workload=replace(frozen.workload, batch_values=(batch,)),
        )
        rows = run_sweep(candidate)
        if len(rows) == 1 and rows[0].capacity_feasible:
            return candidate
    raise ValueError("sensitivity has no capacity-feasible batch baseline")


def metric_range(
    values: Iterable[float | None], baseline: float | None
) -> tuple[float | None, float | None, float | None]:
    """Return min, max, and (max-min)/baseline over feasible values."""
    feasible_values = tuple(value for value in values if value is not None)
    if not feasible_values:
        return None, None, None
    minimum = min(feasible_values)
    maximum = max(feasible_values)
    relative_range = None if baseline == 0.0 else (maximum - minimum) / baseline
    return minimum, maximum, relative_range


def _axis_values(
    config: SimulationConfig,
) -> dict[str, tuple[float | int | str, ...]]:
    return {
        "r": tuple(config.r_values),
        "r_remote": tuple(config.workload.r_remote_values),
        "comm_mode": COMM_MODE_VALUES,
        "thermal_scenario": THERMAL_SCENARIO_VALUES,
        "local_hbm_count": tuple(
            range(1, config.channels.local_hbm.max_count + 1)
        ),
        "address_mapping.policy": ADDRESS_MAPPING_VALUES,
        "h2h.lane_rate_gbps": tuple(config.channels.h2h.lane_rate_gbps),
        "h2h.n_pillar": tuple(config.channels.h2h.n_pillar),
        "batch": tuple(config.workload.batch_values),
        "side_hbm.link_energy_pj_per_bit": SIDE_HBM_LINK_ENERGY_VALUES,
        "local_hbm.stack.hbm_idle_power_w": LOCAL_HBM_IDLE_POWER_VALUES,
        "side_hbm_path": tuple(config.side_hbm_path_values),
    }


def _point_config(
    baseline: SimulationConfig,
    axis: str,
    value: float | int | str,
) -> SimulationConfig:
    if axis == "r":
        return replace(baseline, r_values=(float(value),))
    if axis == "r_remote":
        return replace(
            baseline,
            workload=replace(
                baseline.workload, r_remote_values=(float(value),)
            ),
        )
    if axis == "comm_mode":
        return replace(baseline, comm_mode=str(value))
    if axis == "thermal_scenario":
        return replace(baseline, thermal_scenario=str(value))
    if axis == "local_hbm_count":
        return replace(
            baseline,
            channels=replace(
                baseline.channels,
                local_hbm=replace(
                    baseline.channels.local_hbm, count_values=(int(value),)
                ),
            ),
        )
    if axis == "address_mapping.policy":
        policy = str(value)
        weights = HOTSPOT_WEIGHTS if policy == "weighted" else None
        if (
            policy == "weighted"
            and baseline.channels.local_hbm.count_values != (len(HOTSPOT_WEIGHTS),)
        ):
            raise ValueError(
                "intentional weighted hotspot profile requires four local HBM stacks"
            )
        return replace(
            baseline,
            channels=replace(
                baseline.channels,
                local_hbm=replace(
                    baseline.channels.local_hbm,
                    address_mapping=AddressMappingConfig(
                        policy=policy, weights=weights
                    ),
                ),
            ),
        )
    if axis == "h2h.lane_rate_gbps":
        return replace(
            baseline,
            channels=replace(
                baseline.channels,
                h2h=replace(
                    baseline.channels.h2h, lane_rate_gbps=(float(value),)
                ),
            ),
        )
    if axis == "h2h.n_pillar":
        return replace(
            baseline,
            channels=replace(
                baseline.channels,
                h2h=replace(
                    baseline.channels.h2h, n_pillar=(int(value),)
                ),
            ),
        )
    if axis == "batch":
        return replace(
            baseline,
            workload=replace(baseline.workload, batch_values=(int(value),)),
        )
    if axis == "side_hbm.link_energy_pj_per_bit":
        return replace(
            baseline,
            channels=replace(
                baseline.channels,
                side_hbm=replace(
                    baseline.channels.side_hbm,
                    link_energy_pj_per_bit=float(value),
                ),
            ),
        )
    if axis == "local_hbm.stack.hbm_idle_power_w":
        return replace(
            baseline,
            channels=replace(
                baseline.channels,
                local_hbm=replace(
                    baseline.channels.local_hbm,
                    stack=replace(
                        baseline.channels.local_hbm.stack,
                        hbm_idle_power_w=float(value),
                    ),
                ),
            ),
        )
    if axis == "side_hbm_path":
        return replace(baseline, side_hbm_path_values=(str(value),))
    raise ValueError(f"unknown sensitivity axis: {axis}")


def _simulate_one(config: SimulationConfig) -> SweepRow:
    rows = run_sweep(config)
    if len(rows) != 1:
        raise RuntimeError("sensitivity point must expand to exactly one design point")
    return rows[0]


def _baseline_value(config: SimulationConfig, axis: str) -> float | int | str:
    if axis == "r":
        return config.r_values[0]
    if axis == "r_remote":
        return config.workload.r_remote_values[0]
    if axis == "comm_mode":
        return config.comm_mode
    if axis == "thermal_scenario":
        return config.thermal_scenario
    if axis == "local_hbm_count":
        return config.channels.local_hbm.count_values[0]
    if axis == "address_mapping.policy":
        return config.channels.local_hbm.address_mapping.policy
    if axis == "h2h.lane_rate_gbps":
        return config.channels.h2h.lane_rate_gbps[0]
    if axis == "h2h.n_pillar":
        return config.channels.h2h.n_pillar[0]
    if axis == "batch":
        return config.workload.batch_values[0]
    if axis == "side_hbm.link_energy_pj_per_bit":
        return config.channels.side_hbm.link_energy_pj_per_bit
    if axis == "local_hbm.stack.hbm_idle_power_w":
        return config.channels.local_hbm.stack.hbm_idle_power_w
    if axis == "side_hbm_path":
        return config.side_hbm_path_values[0]
    raise ValueError(f"unknown sensitivity axis: {axis}")


def _flips(values: Iterable[bool]) -> bool:
    return set(values) == {False, True}


def _axis_notes(axis: str) -> str:
    notes = []
    if axis not in ARCHITECTURE_APPLICABLE_AXES:
        notes.append(
            "architecture feasibility does not model this axis; False is "
            "structural, not an empirical no-impact result"
        )
    if axis == "r":
        notes.append(
            "native r=0 shadow evaluation removes the actual OAT r variation, "
            "so an architecture-feasibility flip is not expected"
        )
    if axis == "address_mapping.policy":
        notes.append(
            "weighted uses an intentional worst-case hotspot profile "
            "[0.7, 0.1, 0.1, 0.1], not an average weighted distribution"
        )
    if axis == "side_hbm.link_energy_pj_per_bit":
        notes.append(
            "energy-only literature range: 0.54 pJ/bit on-package and "
            "10/20/40 pJ/bit on-board sensitivity references, not measured "
            "uHBM values; this term is intentionally zero under the "
            "shared_h2h production baseline because traversed edge bytes are "
            "charged to h2h_energy_j, and only affects independent mode"
        )
    if axis == "local_hbm.stack.hbm_idle_power_w":
        notes.append(
            "energy-only literature range: 0.4 W is a Stow Fig. 3(a) visual "
            "read; 5.2 W is a cross-methodology high reference, not an upper "
            "bound"
        )
    if axis == "side_hbm_path":
        notes.append(
            "research bounding axis: shared_h2h routes remote-read payloads "
            "over H2H while independent preserves the legacy separate path; "
            "architecture feasibility is inapplicable because the analytical "
            "h2h_feasibility() model omits side traffic"
        )
    return "; ".join(notes)


def run_sensitivity(config: SimulationConfig) -> list[SensitivityRow]:
    """Run each configured axis one-at-a-time around a feasible baseline."""
    baseline = build_baseline_config(config)
    baseline_result = _simulate_one(baseline)
    if (
        not baseline_result.capacity_feasible
        or baseline_result.tpot_s is None
        or baseline_result.communication_time_s is None
    ):
        raise ValueError("sensitivity baseline must be capacity feasible")

    rows = []
    for axis, values in _axis_values(config).items():
        point_rows = []
        architecture_states = []
        for value in values:
            point_config = _point_config(baseline, axis, value)
            point_rows.append(_simulate_one(point_config))

            # h2h_feasibility() is natively exposed by run_sweep only at r=0.
            # This shadow evaluation deliberately ignores the point's actual r.
            shadow_config = replace(point_config, r_values=(0.0,))
            shadow_row = _simulate_one(shadow_config)
            if shadow_row.architecture_feasible is None:
                raise RuntimeError("r=0 shadow must report architecture feasibility")
            architecture_states.append(shadow_row.architecture_feasible)

        tpot_min, tpot_max, tpot_relative = metric_range(
            (row.tpot_s for row in point_rows), baseline_result.tpot_s
        )
        comm_min, comm_max, comm_relative = metric_range(
            (row.communication_time_s for row in point_rows),
            baseline_result.communication_time_s,
        )
        energy_min, energy_max, energy_relative = metric_range(
            (row.energy_per_token_j for row in point_rows),
            baseline_result.energy_per_token_j,
        )
        throughput_min, throughput_max, throughput_relative = metric_range(
            (row.decode_throughput_tokens_per_s for row in point_rows),
            baseline_result.decode_throughput_tokens_per_s,
        )
        capacity_states = [row.capacity_feasible for row in point_rows]
        n_infeasible = sum(not state for state in capacity_states)
        rows.append(
            SensitivityRow(
                axis=axis,
                baseline_value=str(_baseline_value(baseline, axis)),
                swept_values=json.dumps(list(values), separators=(",", ":")),
                n_swept_points=len(values),
                tpot_s_min=tpot_min,
                tpot_s_max=tpot_max,
                tpot_s_relative_range=tpot_relative,
                communication_time_s_min=comm_min,
                communication_time_s_max=comm_max,
                communication_time_s_relative_range=comm_relative,
                energy_per_token_j_min=energy_min,
                energy_per_token_j_max=energy_max,
                energy_per_token_j_relative_range=energy_relative,
                decode_throughput_tokens_per_s_min=throughput_min,
                decode_throughput_tokens_per_s_max=throughput_max,
                decode_throughput_tokens_per_s_relative_range=(
                    throughput_relative
                ),
                flips_capacity_feasible=_flips(capacity_states),
                flips_architecture_feasible=(
                    _flips(architecture_states)
                    if axis in ARCHITECTURE_APPLICABLE_AXES
                    else False
                ),
                n_capacity_infeasible_points=n_infeasible,
                capacity_infeasible_fraction=n_infeasible / len(values),
                architecture_feasibility_applicable=(
                    axis in ARCHITECTURE_APPLICABLE_AXES
                ),
                architecture_feasibility_semantics=(
                    ARCHITECTURE_FEASIBILITY_SEMANTICS
                ),
                notes=_axis_notes(axis),
            )
        )
    return rows


def write_sensitivity_csv(
    rows: list[SensitivityRow], path: str | Path
) -> None:
    """Write one sensitivity-summary row per OAT axis."""
    if not rows:
        raise ValueError("cannot write an empty sensitivity report")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="uHBM Stage 2 one-at-a-time sensitivity analysis"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage2_sensitivity.csv"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = run_sensitivity(load_config(args.config))
    write_sensitivity_csv(rows, args.output)
    print(f"Analyzed {len(rows)} sensitivity axes; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
