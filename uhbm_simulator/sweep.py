"""Configuration sweeps and CSV output for Stage 2 experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from stage1_model import BW_LOCAL_0

from uhbm_simulator.analysis import h2h_feasibility
from uhbm_simulator.config import (
    DEFAULT_H2H_LANE_COUNT,
    SimulationConfig,
    h2h_bandwidth_bps,
)
from uhbm_simulator.simulator import (
    LocalHBMCapacityError,
    SimulationParameters,
    calculate_channel_energy,
    run_simulation,
)
from uhbm_simulator.workload import build_decode_signature, build_prefill_signature


# NVIDIA H100 SXM5 FP16 Tensor Core dense peak = 990 TFLOPS (NVIDIA H100 Tensor Core
# GPU datasheet). 기존 1 PFLOP/s는 반올림된 임의값이었음 -- 실측 GPU 스펙으로 교체.
# TODO(태현): 실제 target GPU가 H100이 아니라면(예: B200/H200) 해당 스펙으로 교체.
DEFAULT_GPU_PEAK_FLOPS = 990e12
# TODO(태현): 근거 없음 -- base-die endpoint 회로가 아직 없어 공개 문헌으로 특정 불가.
# UCIe Advanced Package급 die-to-die PHY latency가 통상 수 ns 수준이라는 점과 자릿수는
# 맞으나(경험적 정합성일 뿐 직접 인용은 아님), Stage 3 RTL 결과가 나오면 교체할 것.
DEFAULT_BASE_DIE_LATENCY_S = 5e-9


@dataclass(frozen=True)
class SweepRow:
    baseline: str
    side_hbm_path: str
    side_hbm_owner_scope: str
    pillar_operating_point: str | None
    pillar_link_bw_bps: float | None
    pillar_link_latency_s: float | None
    pillar_link_energy_pj_per_bit: float | None
    comm_mode: str
    thermal_scenario: str
    phase: str
    model: str
    tp: int
    pp: int
    seq_len: int
    batch: int
    r: float
    r_remote: float
    local_hbm_count: int
    lane_rate_gbps: float
    n_pillar: int
    h2h_bw_bps: float
    h2h_feasibility_checked: bool
    required_h2h_bw_bps: float | None
    required_h2h_latency_s: float | None
    h2h_bw_feasible: bool | None
    h2h_latency_feasible: bool | None
    architecture_feasible: bool | None
    architecture_feasibility_applicable: bool
    capacity_feasible: bool
    ttft_s: float | None
    tpot_s: float | None
    communication_time_s: float | None
    overlap_s: float | None
    local_hbm_utilization: float | None
    local_hbm_utilization_max: float | None
    side_hbm_utilization: float | None
    h2h_utilization: float | None
    pillar_link_utilization: float | None
    gpu_direct_utilization: float | None
    gpu_compute_utilization: float | None
    total_queue_time_s: float | None
    collective_h2h_edge_bytes: float | None
    side_hbm_read_h2h_edge_bytes: float | None
    pillar_side_read_edge_bytes: float | None
    local_hbm_energy_j: float | None
    local_hbm_idle_energy_j: float | None
    side_hbm_energy_j: float | None
    side_hbm_link_energy_j: float | None
    h2h_energy_j: float | None
    pillar_link_energy_j: float | None
    gpu_direct_energy_j: float | None
    total_energy_j: float | None
    decode_throughput_tokens_per_s: float | None
    prefill_throughput_tokens_per_s: float | None
    energy_per_token_j: float | None


# TODO(태현): B0/B1/B2의 numerical parameter sets가 정해지면 baseline label을
# 실제 parameter preset 선택에 연결한다. v3에서는 r/comm_mode가 동작을 결정한다.


def _average_utilization(metrics, prefix: str) -> float:
    values = [
        metric.utilization
        for name, metric in metrics.items()
        if name.startswith(prefix)
    ]
    return sum(values) / len(values) if values else 0.0


def _max_utilization(metrics, prefix: str) -> float:
    values = [
        metric.utilization
        for name, metric in metrics.items()
        if name.startswith(prefix)
    ]
    return max(values, default=0.0)


def _build_signature(config: SimulationConfig, batch: int):
    workload = config.workload
    builder = (
        build_prefill_signature if workload.phase == "prefill" else build_decode_signature
    )
    return builder(
        workload.model,
        tp=workload.tp,
        seq_len=workload.seq_len,
        batch=batch,
    )


def run_sweep(config: SimulationConfig) -> list[SweepRow]:
    """Expand every YAML sweep axis and simulate each design point."""
    workload = config.workload
    channels = config.channels
    rows = []
    route_points = []
    for side_hbm_path in config.side_hbm_path_values:
        for owner_scope in config.side_hbm_owner_scope_values:
            if side_hbm_path == "pillar_bypass":
                pillar = channels.pillar_link
                if pillar is None:
                    raise ValueError("pillar_bypass requires pillar_link config")
                route_points.extend(
                    (side_hbm_path, owner_scope, point)
                    for point in pillar.operating_points
                )
            else:
                route_points.append((side_hbm_path, owner_scope, None))
    for (
        r,
        r_remote,
        batch,
        local_hbm_count,
        lane_rate,
        n_pillar,
        route_point,
    ) in product(
        config.r_values,
        workload.r_remote_values,
        workload.batch_values,
        channels.local_hbm.count_values,
        channels.h2h.lane_rate_gbps,
        channels.h2h.n_pillar,
        route_points,
    ):
        side_hbm_path, owner_scope, pillar_point = route_point
        signature = _build_signature(config, batch)
        h2h_bw = h2h_bandwidth_bps(
            lane_rate,
            n_pillar,
            channels.h2h.n_pillar_max,
            DEFAULT_H2H_LANE_COUNT,
        )
        parameters = SimulationParameters(
            tp=workload.tp,
            r=r,
            r_remote=r_remote,
            comm_mode=config.comm_mode,
            thermal_scenario=config.thermal_scenario,
            local_hbm_stack_count=local_hbm_count,
            local_hbm_stack_bw_bps=channels.local_hbm.stack.bw_bps,
            local_hbm_stack_capacity_bytes=(
                channels.local_hbm.stack.capacity_bytes
            ),
            local_hbm_mapping_policy=(
                channels.local_hbm.address_mapping.policy
            ),
            local_hbm_mapping_weights=(
                channels.local_hbm.address_mapping.weights
            ),
            side_hbm_bw_bps=channels.side_hbm.bw_bps,
            side_hbm_latency_s=channels.side_hbm.latency_s,
            h2h_bw_bps=h2h_bw,
            h2h_latency_s=channels.h2h.latency_s,
            gpu_direct_bw_bps=channels.gpu_direct.bw_bps,
            gpu_direct_latency_s=channels.gpu_direct.latency_s,
            gpu_peak_flops=DEFAULT_GPU_PEAK_FLOPS,
            base_die_latency_s=DEFAULT_BASE_DIE_LATENCY_S,
            side_hbm_path=side_hbm_path,
            side_hbm_owner_scope=owner_scope,
            physical_adjacency=(
                config.physical_adjacency_by_tp[workload.tp]
                if owner_scope == "adjacent"
                and config.physical_adjacency_by_tp is not None
                else None
            ),
            pillar_link_bw_bps=(
                pillar_point.bw_bps if pillar_point is not None else None
            ),
            pillar_link_latency_s=(
                pillar_point.latency_s if pillar_point is not None else None
            ),
        )
        feasibility = (
            h2h_feasibility(
                model_name=workload.model,
                tp=workload.tp,
                seq_len=workload.seq_len,
                k_b=(
                    channels.local_hbm.stack.bw_bps
                    * local_hbm_count
                    / BW_LOCAL_0
                ),
                batch=batch,
                actual_bw_bps=h2h_bw,
                actual_h2h_round_latency_s=(
                    channels.h2h.latency_s + DEFAULT_BASE_DIE_LATENCY_S
                ),
                signature=signature,
            )
            if r == 0.0
            else None
        )
        try:
            result = run_simulation(signature, parameters)
        except LocalHBMCapacityError:
            result = None
        metrics = result.resource_metrics if result is not None else None
        energy = (
            calculate_channel_energy(
                metrics,
                channels,
                lane_rate,
                local_hbm_stack_count=local_hbm_count,
                elapsed_s=result.elapsed_s,
                tp=workload.tp,
                side_hbm_path=side_hbm_path,
                pillar_link_energy_pj_per_bit=(
                    pillar_point.energy_pj_per_bit
                    if pillar_point is not None
                    else None
                ),
            )
            if metrics is not None
            else None
        )
        rows.append(
            SweepRow(
                baseline=config.baseline,
                side_hbm_path=side_hbm_path,
                side_hbm_owner_scope=owner_scope,
                pillar_operating_point=(
                    pillar_point.name if pillar_point is not None else None
                ),
                pillar_link_bw_bps=(
                    pillar_point.bw_bps if pillar_point is not None else None
                ),
                pillar_link_latency_s=(
                    pillar_point.latency_s if pillar_point is not None else None
                ),
                pillar_link_energy_pj_per_bit=(
                    pillar_point.energy_pj_per_bit
                    if pillar_point is not None
                    else None
                ),
                comm_mode=config.comm_mode,
                thermal_scenario=config.thermal_scenario,
                phase=workload.phase,
                model=workload.model,
                tp=workload.tp,
                pp=workload.pp,
                seq_len=workload.seq_len,
                batch=batch,
                r=r,
                r_remote=r_remote,
                local_hbm_count=local_hbm_count,
                lane_rate_gbps=lane_rate,
                n_pillar=n_pillar,
                h2h_bw_bps=h2h_bw,
                h2h_feasibility_checked=feasibility is not None,
                required_h2h_bw_bps=(
                    feasibility.required_bw_bps if feasibility else None
                ),
                required_h2h_latency_s=(
                    feasibility.required_latency_s if feasibility else None
                ),
                h2h_bw_feasible=(
                    feasibility.bandwidth_feasible if feasibility else None
                ),
                h2h_latency_feasible=(
                    feasibility.latency_feasible if feasibility else None
                ),
                architecture_feasible=(
                    feasibility.architecture_feasible if feasibility else None
                ),
                architecture_feasibility_applicable=(
                    side_hbm_path == "independent"
                ),
                capacity_feasible=result is not None,
                ttft_s=result.ttft_s if result is not None else None,
                tpot_s=result.tpot_s if result is not None else None,
                communication_time_s=(
                    result.communication_time_s if result is not None else None
                ),
                overlap_s=result.overlap_s if result is not None else None,
                local_hbm_utilization=(
                    _average_utilization(metrics, "local_hbm")
                    if metrics is not None
                    else None
                ),
                local_hbm_utilization_max=(
                    _max_utilization(metrics, "local_hbm")
                    if metrics is not None
                    else None
                ),
                side_hbm_utilization=(
                    _average_utilization(metrics, "side_hbm")
                    if metrics is not None
                    else None
                ),
                h2h_utilization=(
                    _average_utilization(metrics, "h2h_link")
                    if metrics is not None
                    else None
                ),
                pillar_link_utilization=(
                    _average_utilization(metrics, "pillar_link")
                    if metrics is not None
                    else None
                ),
                gpu_direct_utilization=(
                    _average_utilization(metrics, "gpu_direct_link")
                    if metrics is not None
                    else None
                ),
                gpu_compute_utilization=(
                    _average_utilization(metrics, "gpu_compute")
                    if metrics is not None
                    else None
                ),
                total_queue_time_s=(
                    sum(metric.total_queue_time_s for metric in metrics.values())
                    if metrics is not None
                    else None
                ),
                collective_h2h_edge_bytes=(
                    result.collective_h2h_edge_bytes
                    if result is not None
                    else None
                ),
                side_hbm_read_h2h_edge_bytes=(
                    result.side_hbm_read_h2h_edge_bytes
                    if result is not None
                    else None
                ),
                pillar_side_read_edge_bytes=(
                    result.pillar_side_read_edge_bytes
                    if result is not None
                    else None
                ),
                local_hbm_energy_j=(
                    energy.local_hbm_energy_j if energy is not None else None
                ),
                local_hbm_idle_energy_j=(
                    energy.local_hbm_idle_energy_j if energy is not None else None
                ),
                side_hbm_energy_j=(
                    energy.side_hbm_energy_j if energy is not None else None
                ),
                side_hbm_link_energy_j=(
                    energy.side_hbm_link_energy_j if energy is not None else None
                ),
                h2h_energy_j=(energy.h2h_energy_j if energy is not None else None),
                pillar_link_energy_j=(
                    energy.pillar_link_energy_j if energy is not None else None
                ),
                gpu_direct_energy_j=(
                    energy.gpu_direct_energy_j if energy is not None else None
                ),
                total_energy_j=(energy.total_energy_j if energy is not None else None),
                decode_throughput_tokens_per_s=(
                    batch / result.tpot_s
                    if result is not None and workload.phase == "decode"
                    else None
                ),
                prefill_throughput_tokens_per_s=(
                    batch * workload.seq_len / result.ttft_s
                    if result is not None and workload.phase == "prefill"
                    else None
                ),
                energy_per_token_j=(
                    energy.total_energy_j
                    / (
                        batch
                        if workload.phase == "decode"
                        else batch * workload.seq_len
                    )
                    if energy is not None
                    else None
                ),
            )
        )
    return rows


def write_csv(rows: list[SweepRow], path: str | Path) -> None:
    """Write sweep rows to a stable UTF-8 CSV schema."""
    if not rows:
        raise ValueError("cannot write an empty sweep")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
