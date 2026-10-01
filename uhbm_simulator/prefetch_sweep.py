"""Metrics, Cartesian sweeps, CSV output, and CLI for layer prefetch studies."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from itertools import product
import json
import math
from pathlib import Path
import statistics

from .prefetch_config import (
    CapacityMemoryConfig,
    PrefetchSimulationConfig,
    load_prefetch_config,
)
from .prefetch_simulator import PrefetchSimulationResult, run_prefetch_simulation
from .resources import TrafficClass


@dataclass(frozen=True)
class PrefetchPathMetric:
    transferred_bytes: float
    request_count: int
    queue_time_s: float
    service_time_s: float
    latency_time_s: float
    energy_j: float


@dataclass(frozen=True)
class PrefetchMetrics:
    paths: dict[str, PrefetchPathMetric]
    energy_by_domain_j: dict[str, float]
    total_dynamic_energy_j: float
    local_lane_utilization_mean: float
    local_lane_utilization_max: float
    critical_local_lane: str | None


@dataclass(frozen=True)
class PrefetchSweepRow:
    architecture: str
    architecture_kind: str
    capacity_memory_profile: str
    run_mode: str
    start_state: str
    model: str
    tp: int
    seq_len: int
    batch: int
    decode_tokens: int
    weight_residency_policy: str
    effective_weight_policy: str | None
    weight_chunk_bytes: int
    kv_chunk_bytes: int
    kv_writeback_chunk_bytes: int
    max_inflight_chunks_per_lane: int | None
    capacity_feasible: bool
    infeasibility_reason: str | None
    first_infeasible_token: int | None
    elapsed_s: float | None
    ttft_s: float | None
    first_token_latency_s: float | None
    tpot_s: float | None
    token_latency_mean_s: float | None
    token_latency_median_s: float | None
    token_latency_max_s: float | None
    steady_token_latency_mean_s: float | None
    decode_throughput_tokens_per_s: float | None
    prefetch_stall_total_s: float | None
    weight_prefetch_stall_s: float | None
    kv_prefetch_stall_s: float | None
    hidden_prefetch_fraction: float | None
    prefetch_deadline_misses: int | None
    prefetch_limiting_class: str | None
    kv_writeback_deadline_misses: int | None
    kv_writeback_max_lateness_s: float | None
    total_dynamic_energy_j: float | None
    local_lane_utilization_mean: float | None
    local_lane_utilization_max: float | None
    critical_local_lane: str | None
    path_metrics_json: str | None


_CAPACITY_DEVICE_CLASSES = {TrafficClass.CAPACITY_MEMORY_READ}
_CAPACITY_LINK_CLASSES = {
    TrafficClass.CAPACITY_TO_UHBM,
    TrafficClass.CAPACITY_TO_2_5D_HBM,
}
_LOCAL_MEMORY_CLASSES = {
    TrafficClass.UHBM_GPU_READ,
    TrafficClass.GPU_UHBM_WRITE,
    TrafficClass.UHBM_PREFETCH_WRITE,
    TrafficClass.UHBM_WRITEBACK_READ,
    TrafficClass.HBM_2_5D_GPU_READ,
    TrafficClass.GPU_HBM_2_5D_WRITE,
    TrafficClass.HBM_2_5D_PREFETCH_WRITE,
    TrafficClass.COLLECTIVE_SOURCE_READ,
    TrafficClass.COLLECTIVE_FINAL_WRITE,
    TrafficClass.COLLECTIVE_FINAL_REREAD,
}
_SIDE_HBM_CLASSES = {
    TrafficClass.SIDE_HBM_READ,
    TrafficClass.SIDE_HBM_WRITE,
}
_SIDE_LINK_CLASSES = {
    TrafficClass.SIDE_TO_UHBM,
    TrafficClass.UHBM_TO_SIDE,
}
_COLLECTIVE_TRANSPORT_CLASSES = {
    TrafficClass.COLLECTIVE_H2H_TRANSPORT,
    TrafficClass.COLLECTIVE_GPU_DIRECT_TRANSPORT,
}


def _energy_parameters(config: PrefetchSimulationConfig) -> dict[str, tuple[str, float]]:
    architecture = config.architectures[0]
    mapping = {
        traffic_class: (
            "capacity_memory",
            config.capacity_memory.device.read_energy_pj_per_bit,
        )
        for traffic_class in _CAPACITY_DEVICE_CLASSES
    }
    mapping.update(
        {
            traffic_class: (
                "capacity_link",
                config.capacity_memory.link.energy_pj_per_bit,
            )
            for traffic_class in _CAPACITY_LINK_CLASSES
        }
    )
    mapping.update(
        {
            traffic_class: (
                "local_memory",
                architecture.local_memory.energy_pj_per_bit,
            )
            for traffic_class in _LOCAL_MEMORY_CLASSES
        }
    )
    if architecture.side_hbm is not None and architecture.side_link is not None:
        mapping.update(
            {
                traffic_class: (
                    "side_hbm",
                    architecture.side_hbm.energy_pj_per_bit,
                )
                for traffic_class in _SIDE_HBM_CLASSES
            }
        )
        mapping.update(
            {
                traffic_class: (
                    "side_link",
                    architecture.side_link.energy_pj_per_bit,
                )
                for traffic_class in _SIDE_LINK_CLASSES
            }
        )
    mapping[TrafficClass.COLLECTIVE_H2H_TRANSPORT] = (
        "collective_h2h",
        config.execution.h2h_energy_pj_per_bit,
    )
    mapping[TrafficClass.COLLECTIVE_GPU_DIRECT_TRANSPORT] = (
        "collective_gpu_direct",
        config.execution.gpu_direct_energy_pj_per_bit,
    )
    return mapping


def _reported_traffic_class(resource_name: str, traffic_class: str) -> str | None:
    if traffic_class != TrafficClass.UNCLASSIFIED:
        return traffic_class
    if resource_name.startswith("prefetch_collective_h2h["):
        return TrafficClass.COLLECTIVE_H2H_TRANSPORT
    if resource_name.startswith("prefetch_gpu_direct["):
        return TrafficClass.COLLECTIVE_GPU_DIRECT_TRANSPORT
    return None


def calculate_prefetch_metrics(
    config: PrefetchSimulationConfig,
    result: PrefetchSimulationResult,
) -> PrefetchMetrics:
    """Aggregate immutable tagged resource stats without utilization inference."""
    if not result.feasible or result.elapsed_s is None:
        return PrefetchMetrics({}, {}, 0.0, 0.0, 0.0, None)

    aggregated: dict[str, dict[str, list[float] | int]] = {}
    for resource in result.resources:
        for traffic_class, stats in resource.stats.by_traffic_class.items():
            reported_class = _reported_traffic_class(resource.name, traffic_class)
            if reported_class is None:
                continue
            values = aggregated.setdefault(
                reported_class,
                {
                    "bytes": [],
                    "requests": 0,
                    "queue": [],
                    "service": [],
                    "latency": [],
                },
            )
            values["bytes"].append(stats.transferred_amount)
            values["requests"] += stats.request_count
            values["queue"].append(stats.total_queue_time_s)
            values["service"].append(stats.total_service_time_s)
            values["latency"].append(stats.total_latency_time_s)

    energy_parameters = _energy_parameters(config)
    domains: dict[str, list[float]] = {}
    paths = {}
    for traffic_class, values in aggregated.items():
        transferred_bytes = math.fsum(values["bytes"])
        domain, energy_pj_per_bit = energy_parameters.get(
            traffic_class, ("unmodeled", 0.0)
        )
        energy_j = transferred_bytes * 8.0 * energy_pj_per_bit * 1e-12
        domains.setdefault(domain, []).append(energy_j)
        paths[traffic_class] = PrefetchPathMetric(
            transferred_bytes=transferred_bytes,
            request_count=values["requests"],
            queue_time_s=math.fsum(values["queue"]),
            service_time_s=math.fsum(values["service"]),
            latency_time_s=math.fsum(values["latency"]),
            energy_j=energy_j,
        )

    domain_totals = {
        domain: math.fsum(energies) for domain, energies in domains.items()
    }
    local_resources = [
        resource
        for resource in result.resources
        if ".uhbm[" in resource.name or ".hbm_2_5d[" in resource.name
    ]
    utilizations = [
        resource.stats.utilization(result.elapsed_s) for resource in local_resources
    ]
    if local_resources:
        critical_index = max(
            range(len(local_resources)), key=lambda index: utilizations[index]
        )
        critical_lane = local_resources[critical_index].name
        utilization_mean = math.fsum(utilizations) / len(utilizations)
        utilization_max = utilizations[critical_index]
    else:
        critical_lane = None
        utilization_mean = 0.0
        utilization_max = 0.0

    return PrefetchMetrics(
        paths=paths,
        energy_by_domain_j=domain_totals,
        total_dynamic_energy_j=math.fsum(
            metric.energy_j for metric in paths.values()
        ),
        local_lane_utilization_mean=utilization_mean,
        local_lane_utilization_max=utilization_max,
        critical_local_lane=critical_lane,
    )


def _capacity_profiles(
    config: PrefetchSimulationConfig,
) -> tuple[CapacityMemoryConfig, ...]:
    value = config.capacity_memory
    return value if isinstance(value, tuple) else (value,)


def expand_prefetch_points(
    config: PrefetchSimulationConfig,
) -> tuple[PrefetchSimulationConfig, ...]:
    """Return the full Cartesian product as singleton executable configs."""
    points = []
    axes = product(
        config.architectures,
        _capacity_profiles(config),
        config.workload.run_mode_values,
        config.workload.start_state_values,
        config.workload.batch_values,
        config.workload.decode_tokens_values,
        config.prefetch.weight_residency_policy_values,
        config.prefetch.weight_chunk_bytes_values,
        config.prefetch.kv_chunk_bytes_values,
        config.prefetch.kv_writeback_chunk_bytes_values,
        config.prefetch.max_inflight_chunks_per_lane_values,
    )
    for (
        architecture,
        capacity_memory,
        run_mode,
        start_state,
        batch,
        decode_tokens,
        policy,
        weight_chunk,
        kv_chunk,
        writeback_chunk,
        inflight,
    ) in axes:
        points.append(
            replace(
                config,
                architectures=(architecture,),
                capacity_memory=capacity_memory,
                workload=replace(
                    config.workload,
                    run_mode_values=(run_mode,),
                    start_state_values=(start_state,),
                    batch_values=(batch,),
                    decode_tokens_values=(decode_tokens,),
                ),
                prefetch=replace(
                    config.prefetch,
                    weight_residency_policy_values=(policy,),
                    weight_chunk_bytes_values=(weight_chunk,),
                    kv_chunk_bytes_values=(kv_chunk,),
                    kv_writeback_chunk_bytes_values=(writeback_chunk,),
                    max_inflight_chunks_per_lane_values=(inflight,),
                ),
            )
        )
    return tuple(points)


def _path_metrics_json(metrics: PrefetchMetrics) -> str:
    return json.dumps(
        {name: asdict(path) for name, path in sorted(metrics.paths.items())},
        sort_keys=True,
        separators=(",", ":"),
    )


def _sweep_row(
    point: PrefetchSimulationConfig,
    result: PrefetchSimulationResult,
) -> PrefetchSweepRow:
    architecture = point.architectures[0]
    capacity_memory = point.capacity_memory
    assert isinstance(capacity_memory, CapacityMemoryConfig)
    run_mode = point.workload.run_mode_values[0]
    decode_tokens = point.workload.decode_tokens_values[0]
    metrics = calculate_prefetch_metrics(point, result) if result.feasible else None
    decode_throughput = (
        point.workload.batch_values[0] / result.tpot_s
        if result.tpot_s is not None and result.tpot_s > 0.0
        else None
    )
    token_latencies = result.token_latencies_s
    token_mean = (
        math.fsum(token_latencies) / len(token_latencies)
        if token_latencies
        else None
    )
    token_median = statistics.median(token_latencies) if token_latencies else None
    token_max = max(token_latencies) if token_latencies else None
    measured_layers = (
        tuple(
            timing
            for timing in result.trace.layer_timings
            if timing.token_index >= 0
        )
        if result.trace is not None
        else ()
    )
    if measured_layers:
        total_stall = math.fsum(
            timing.prefetch_stall_s for timing in measured_layers
        )
        weight_stall = math.fsum(
            max(0.0, max(timing.weight_lane_ready_s) - timing.previous_layer_end_s)
            for timing in measured_layers
        )
        kv_stall = math.fsum(
            max(0.0, max(timing.kv_lane_ready_s) - timing.previous_layer_end_s)
            for timing in measured_layers
        )
        total_prefetch_time = math.fsum(
            max(0.0, timing.layer_ready_s - timing.prefetch_launch_s)
            for timing in measured_layers
        )
        hidden_fraction = (
            min(1.0, max(0.0, 1.0 - total_stall / total_prefetch_time))
            if total_prefetch_time > 0.0
            else 1.0
        )
        deadline_misses = sum(
            timing.prefetch_stall_s > 0.0 for timing in measured_layers
        )
        if weight_stall > kv_stall:
            limiting_class = "weight"
        elif kv_stall > weight_stall:
            limiting_class = "kv"
        elif weight_stall > 0.0:
            limiting_class = "tie"
        else:
            limiting_class = None

        next_launch = {
            (timing.token_index, timing.layer_index): timing.prefetch_launch_s
            for timing in measured_layers
        }
        writeback_lateness = []
        for timing in measured_layers:
            deadline = next_launch.get((timing.token_index + 1, timing.layer_index))
            if deadline is not None and timing.writeback_end_s is not None:
                writeback_lateness.append(max(0.0, timing.writeback_end_s - deadline))
        writeback_misses = sum(value > 0.0 for value in writeback_lateness)
        writeback_max_lateness = max(writeback_lateness, default=0.0)
    else:
        total_stall = weight_stall = kv_stall = hidden_fraction = None
        deadline_misses = None
        limiting_class = None
        writeback_misses = None
        writeback_max_lateness = None
    reason = result.infeasibility_reason
    return PrefetchSweepRow(
        architecture=architecture.name,
        architecture_kind=architecture.kind,
        capacity_memory_profile=capacity_memory.name,
        run_mode=run_mode,
        start_state=point.workload.start_state_values[0],
        model=point.workload.model,
        tp=point.workload.tp,
        seq_len=point.workload.seq_len,
        batch=point.workload.batch_values[0],
        decode_tokens=decode_tokens,
        weight_residency_policy=point.prefetch.weight_residency_policy_values[0],
        effective_weight_policy=result.residency.effective_policy,
        weight_chunk_bytes=point.prefetch.weight_chunk_bytes_values[0],
        kv_chunk_bytes=point.prefetch.kv_chunk_bytes_values[0],
        kv_writeback_chunk_bytes=point.prefetch.kv_writeback_chunk_bytes_values[0],
        max_inflight_chunks_per_lane=(
            point.prefetch.max_inflight_chunks_per_lane_values[0]
        ),
        capacity_feasible=result.feasible,
        infeasibility_reason=(reason.value if reason is not None else None),
        first_infeasible_token=result.first_infeasible_token,
        elapsed_s=result.elapsed_s,
        ttft_s=result.ttft_s,
        first_token_latency_s=result.first_token_latency_s,
        tpot_s=result.tpot_s,
        token_latency_mean_s=token_mean,
        token_latency_median_s=token_median,
        token_latency_max_s=token_max,
        steady_token_latency_mean_s=(
            token_mean
            if point.workload.start_state_values[0] == "steady_state"
            else None
        ),
        decode_throughput_tokens_per_s=decode_throughput,
        prefetch_stall_total_s=total_stall,
        weight_prefetch_stall_s=weight_stall,
        kv_prefetch_stall_s=kv_stall,
        hidden_prefetch_fraction=hidden_fraction,
        prefetch_deadline_misses=deadline_misses,
        prefetch_limiting_class=limiting_class,
        kv_writeback_deadline_misses=writeback_misses,
        kv_writeback_max_lateness_s=writeback_max_lateness,
        total_dynamic_energy_j=(
            metrics.total_dynamic_energy_j if metrics is not None else None
        ),
        local_lane_utilization_mean=(
            metrics.local_lane_utilization_mean if metrics is not None else None
        ),
        local_lane_utilization_max=(
            metrics.local_lane_utilization_max if metrics is not None else None
        ),
        critical_local_lane=(metrics.critical_local_lane if metrics is not None else None),
        path_metrics_json=(_path_metrics_json(metrics) if metrics is not None else None),
    )


def run_prefetch_sweep(
    config: PrefetchSimulationConfig,
) -> tuple[PrefetchSweepRow, ...]:
    return tuple(
        _sweep_row(point, run_prefetch_simulation(point, detailed_trace=True))
        for point in expand_prefetch_points(config)
    )


def write_prefetch_csv(
    rows: tuple[PrefetchSweepRow, ...] | list[PrefetchSweepRow],
    output: str | Path,
) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PrefetchSweepRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rows = run_prefetch_sweep(load_prefetch_config(args.config))
    write_prefetch_csv(rows, args.output)
    print(f"wrote {len(rows)} prefetch sweep rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
