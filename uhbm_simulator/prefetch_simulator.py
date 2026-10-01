"""Dedicated full-layer-ready prefetch event simulator.

This module is opt-in and does not alter the legacy Stage 2 simulator.  It
combines the eight paired memory lanes with the corrected D-1 global ring
coordinator and D-2 attention/MLP dependency order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

import simpy

from stage1_model import MODELS, attention_parameter_fraction

from .prefetch_config import PrefetchSimulationConfig
from .prefetch_resources import (
    PrefetchResourceLayer,
    build_prefetch_resources,
    transfer_chunks,
)
from .prefetch_workload import (
    InfeasibilityReason,
    LayerFootprint,
    WeightResidencyDecision,
    build_layer_plans,
    decide_weight_residency,
)
from .resources import RateResource, TrafficClass
from .simulator import (
    H2HTrafficCounters,
    RingCollectiveCoordinator,
    SimulationParameters,
)


@dataclass(frozen=True)
class LayerTiming:
    token_index: int
    layer_index: int
    prefetch_launch_s: float
    lane_ready_s: tuple[float, ...]
    weight_lane_ready_s: tuple[float, ...]
    kv_lane_ready_s: tuple[float, ...]
    layer_ready_s: float
    previous_layer_end_s: float
    compute_start_s: float
    attention_compute_start_s: float
    attention_compute_end_s: float
    attention_collective_end_s: float
    mlp_compute_start_s: float
    mlp_compute_end_s: float
    mlp_collective_end_s: float
    compute_end_s: float
    prefetch_slack_s: float
    prefetch_stall_s: float
    writeback_end_s: float | None


@dataclass(frozen=True)
class TokenTiming:
    token_index: int
    phase: str
    started_s: float
    ended_s: float
    latency_s: float
    measured: bool
    kv_resident_bytes_per_lane_start: float
    kv_resident_bytes_per_lane_end: float


@dataclass(frozen=True)
class PrefetchTrace:
    layer_timings: tuple[LayerTiming, ...]
    token_timings: tuple[TokenTiming, ...]


@dataclass(frozen=True)
class _LayerPrefetchReady:
    lane_ready_s: tuple[float, ...]
    weight_lane_ready_s: tuple[float, ...]
    kv_lane_ready_s: tuple[float, ...]


@dataclass(frozen=True)
class PrefetchSimulationResult:
    feasible: bool
    infeasibility_reason: InfeasibilityReason | None
    first_infeasible_token: int | None
    residency: WeightResidencyDecision
    elapsed_s: float | None
    ttft_s: float | None
    first_token_latency_s: float | None
    token_latencies_s: tuple[float, ...]
    tpot_s: float | None
    path_bytes: dict[str, float]
    trace: PrefetchTrace | None
    resources: tuple[RateResource, ...]


def _single(values: tuple, name: str):
    if len(values) != 1:
        raise ValueError(f"{name} must be expanded to one sweep value")
    return values[0]


def _new_rank_resources(
    env: simpy.Environment,
    config: PrefetchSimulationConfig,
) -> tuple[PrefetchResourceLayer, ...]:
    architecture = config.architectures[0]
    result = []
    for rank in range(config.workload.tp):
        layer = build_prefetch_resources(
            env, architecture, config.capacity_memory
        )
        for resource in layer.all_resources():
            resource.name = f"rank[{rank}].{resource.name}"
        result.append(layer)
    return tuple(result)


def _collective_resources(
    env: simpy.Environment,
    config: PrefetchSimulationConfig,
    rank_resources: tuple[PrefetchResourceLayer, ...],
):
    tp = config.workload.tp
    execution = config.execution
    edges = [(rank, (rank + 1) % tp) for rank in range(tp)] if tp > 1 else []
    local_hbm = {
        (rank, lane): resources.local_memory[lane]
        for rank, resources in enumerate(rank_resources)
        for lane in range(8)
    }
    return SimpleNamespace(
        local_hbm=local_hbm,
        side_hbm={},
        h2h_link={
            edge: RateResource(
                env,
                f"prefetch_collective_h2h[{edge[0]},{edge[1]}]",
                execution.h2h_bw_bps,
                latency_s=execution.h2h_latency_s,
            )
            for edge in edges
        },
        pillar_link={},
        gpu_direct_link=(
            {
                edge: RateResource(
                    env,
                    f"prefetch_gpu_direct[{edge[0]},{edge[1]}]",
                    execution.gpu_direct_bw_bps,
                    latency_s=execution.gpu_direct_latency_s,
                )
                for edge in edges
            }
            if execution.r > 0.0
            else {}
        ),
        gpu_compute={
            rank: RateResource(
                env,
                f"prefetch_gpu_compute[{rank}]",
                execution.gpu_peak_flops,
            )
            for rank in range(tp)
        },
    )


def _simulation_parameters(config: PrefetchSimulationConfig) -> SimulationParameters:
    architecture = config.architectures[0]
    execution = config.execution
    return SimulationParameters(
        tp=config.workload.tp,
        r=execution.r,
        r_remote=0.0,
        comm_mode=execution.comm_mode,
        thermal_scenario="no_throttling",
        local_hbm_stack_count=8,
        local_hbm_stack_bw_bps=architecture.local_memory.bw_bps,
        local_hbm_stack_capacity_bytes=architecture.local_memory.capacity_bytes,
        local_hbm_mapping_policy="uniform",
        local_hbm_mapping_weights=None,
        side_hbm_bw_bps=(
            architecture.side_hbm.bw_bps
            if architecture.side_hbm is not None
            else architecture.local_memory.bw_bps
        ),
        side_hbm_latency_s=(
            architecture.side_hbm.latency_s
            if architecture.side_hbm is not None
            else 0.0
        ),
        h2h_bw_bps=execution.h2h_bw_bps,
        h2h_latency_s=execution.h2h_latency_s,
        gpu_direct_bw_bps=execution.gpu_direct_bw_bps,
        gpu_direct_latency_s=execution.gpu_direct_latency_s,
        gpu_peak_flops=execution.gpu_peak_flops,
        base_die_latency_s=execution.base_die_latency_s,
        local_hbm_latency_s=architecture.local_memory.latency_s,
        include_materialized_endpoints=execution.include_materialized_endpoints,
        allow_overlap=True,
        strict_collective_dependency=True,
    )


def _descriptor_pools(
    env: simpy.Environment,
    tp: int,
    depth: int | None,
) -> tuple[tuple[simpy.Resource | None, ...], ...]:
    return tuple(
        tuple(
            None if depth is None else simpy.Resource(env, capacity=depth)
            for _ in range(8)
        )
        for _ in range(tp)
    )


def _transfer(
    env: simpy.Environment,
    *,
    amount: float,
    chunk_bytes: int,
    source: RateResource,
    link: RateResource,
    destination: RateResource,
    tags: tuple[str, str, str],
    pool: simpy.Resource | None,
):
    yield env.process(
        transfer_chunks(
            env,
            total_bytes=amount,
            chunk_bytes=chunk_bytes,
            source=source,
            link=link,
            destination=destination,
            source_traffic_class=tags[0],
            link_traffic_class=tags[1],
            destination_traffic_class=tags[2],
            max_inflight_chunks=None,
            inflight_slots=pool,
        )
    )
    return env.now


def _prefetch_layer(
    env: simpy.Environment,
    plan: LayerFootprint,
    config: PrefetchSimulationConfig,
    rank_resources: tuple[PrefetchResourceLayer, ...],
    pools,
    effective_policy: str,
):
    architecture = config.architectures[0]
    weight_chunk = _single(
        config.prefetch.weight_chunk_bytes_values, "weight_chunk_bytes"
    )
    kv_chunk = _single(config.prefetch.kv_chunk_bytes_values, "kv_chunk_bytes")
    rank_lane_ready = []

    def lane_ready(rank: int, lane: int):
        lane_plan = plan.lanes[lane]
        resources = rank_resources[rank]
        events = []
        launched_at = env.now
        weight_event = None
        kv_event = None
        # Stable tie order is weight before KV for reproducibility. No priority
        # is implied once requests reach the shared FCFS destination resource.
        if effective_policy == "streaming" and lane_plan.weight_bytes > 0.0:
            link_tag = (
                TrafficClass.CAPACITY_TO_UHBM
                if architecture.kind == "b2_uhbm"
                else TrafficClass.CAPACITY_TO_2_5D_HBM
            )
            destination_tag = (
                TrafficClass.UHBM_PREFETCH_WRITE
                if architecture.kind == "b2_uhbm"
                else TrafficClass.HBM_2_5D_PREFETCH_WRITE
            )
            weight_event = env.process(
                    _transfer(
                        env,
                        amount=lane_plan.weight_bytes,
                        chunk_bytes=weight_chunk,
                        source=resources.capacity_memory[lane],
                        link=resources.capacity_link[lane],
                        destination=resources.local_memory[lane],
                        tags=(
                            TrafficClass.CAPACITY_MEMORY_READ,
                            link_tag,
                            destination_tag,
                        ),
                        pool=pools[rank][lane],
                    )
            )
            events.append(weight_event)
        if (
            architecture.kind == "b2_uhbm"
            and lane_plan.historical_kv_bytes > 0.0
        ):
            kv_event = env.process(
                    _transfer(
                        env,
                        amount=lane_plan.historical_kv_bytes,
                        chunk_bytes=kv_chunk,
                        source=resources.side_hbm[lane],
                        link=resources.side_link[lane],
                        destination=resources.local_memory[lane],
                        tags=(
                            TrafficClass.SIDE_HBM_READ,
                            TrafficClass.SIDE_TO_UHBM,
                            TrafficClass.UHBM_PREFETCH_WRITE,
                        ),
                        pool=pools[rank][lane],
                    )
            )
            events.append(kv_event)
        if events:
            yield env.all_of(events)
        return (
            env.now,
            weight_event.value if weight_event is not None else launched_at,
            kv_event.value if kv_event is not None else launched_at,
        )

    for rank in range(config.workload.tp):
        rank_lane_ready.append(
            [env.process(lane_ready(rank, lane)) for lane in range(8)]
        )
    yield env.all_of(
        [event for rank_events in rank_lane_ready for event in rank_events]
    )
    rank_zero = tuple(event.value for event in rank_lane_ready[0])
    return _LayerPrefetchReady(
        lane_ready_s=tuple(value[0] for value in rank_zero),
        weight_lane_ready_s=tuple(value[1] for value in rank_zero),
        kv_lane_ready_s=tuple(value[2] for value in rank_zero),
    )


def _local_access(
    env: simpy.Environment,
    rank_resources: tuple[PrefetchResourceLayer, ...],
    amounts: tuple[float, ...],
    traffic_class: str,
):
    events = [
        env.process(
            resources.local_memory[lane].use(amounts[lane], traffic_class)
        )
        for resources in rank_resources
        for lane in range(8)
        if amounts[lane] > 0.0
    ]
    if events:
        yield env.all_of(events)


def _compute_all_ranks(env, collective_resources, amount: float):
    if amount <= 0.0:
        return
    yield env.all_of(
        [
            env.process(resource.use(amount))
            for resource in collective_resources.gpu_compute.values()
        ]
    )


def _collective(
    env: simpy.Environment,
    coordinator: RingCollectiveCoordinator,
    collective_index: int,
    amount: float,
    r: float,
    tp: int,
):
    if tp <= 1 or amount <= 0.0:
        return
    events = [
        env.process(
            coordinator.participate(
                collective_index=collective_index,
                requester=rank,
                h2h_collective_bytes=(1.0 - r) * amount,
                gpu_direct_collective_bytes=r * amount,
            )
        )
        for rank in range(tp)
    ]
    yield env.all_of(events)


def _writeback_layer(
    env: simpy.Environment,
    plan: LayerFootprint,
    config: PrefetchSimulationConfig,
    rank_resources: tuple[PrefetchResourceLayer, ...],
    pools,
):
    architecture = config.architectures[0]
    chunk = _single(
        config.prefetch.kv_writeback_chunk_bytes_values,
        "kv_writeback_chunk_bytes",
    )
    events = []
    for rank, resources in enumerate(rank_resources):
        for lane, lane_plan in enumerate(plan.lanes):
            if lane_plan.new_kv_bytes <= 0.0:
                continue
            if architecture.kind == "b2_uhbm":
                events.append(
                    env.process(
                        _transfer(
                            env,
                            amount=lane_plan.new_kv_bytes,
                            chunk_bytes=chunk,
                            source=resources.local_memory[lane],
                            link=resources.side_link[lane],
                            destination=resources.side_hbm[lane],
                            tags=(
                                TrafficClass.UHBM_WRITEBACK_READ,
                                TrafficClass.UHBM_TO_SIDE,
                                TrafficClass.SIDE_HBM_WRITE,
                            ),
                            pool=pools[rank][lane],
                        )
                    )
                )
            else:
                events.append(
                    env.process(
                        resources.local_memory[lane].use(
                            lane_plan.new_kv_bytes,
                            TrafficClass.GPU_HBM_2_5D_WRITE,
                        )
                    )
                )
    if events:
        yield env.all_of(events)
    return env.now


def _path_bytes(resources: tuple[RateResource, ...]) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for resource in resources:
        for traffic_class, stats in resource.stats.by_traffic_class.items():
            totals.setdefault(traffic_class, []).append(stats.transferred_amount)
    return {
        traffic_class: math.fsum(amounts)
        for traffic_class, amounts in totals.items()
    }


def run_prefetch_simulation(
    config: PrefetchSimulationConfig,
    *,
    detailed_trace: bool = False,
) -> PrefetchSimulationResult:
    """Run one fully expanded dedicated prefetch design point."""
    if len(config.architectures) != 1:
        raise ValueError("architectures must be expanded to one sweep value")
    batch = _single(config.workload.batch_values, "workload.batch")
    decode_tokens = _single(
        config.workload.decode_tokens_values, "workload.decode_tokens"
    )
    run_mode = _single(config.workload.run_mode_values, "workload.run_mode")
    _single(config.workload.start_state_values, "workload.start_state")
    depth = _single(
        config.prefetch.max_inflight_chunks_per_lane_values,
        "max_inflight_chunks_per_lane",
    )

    residency = decide_weight_residency(
        config.architectures[0], config.workload, config.prefetch
    )
    if not residency.feasible:
        return PrefetchSimulationResult(
            feasible=False,
            infeasibility_reason=residency.reason,
            first_infeasible_token=residency.first_infeasible_token,
            residency=residency,
            elapsed_s=None,
            ttft_s=None,
            first_token_latency_s=None,
            token_latencies_s=(),
            tpot_s=None,
            path_bytes={},
            trace=None,
            resources=(),
        )
    env = simpy.Environment()
    rank_resources = _new_rank_resources(env, config)
    collective_resources = _collective_resources(env, config, rank_resources)
    parameters = _simulation_parameters(config)
    counters = H2HTrafficCounters()
    coordinator = RingCollectiveCoordinator(
        env, parameters, collective_resources, counters
    )
    pools = _descriptor_pools(env, config.workload.tp, depth)
    attention_fraction = attention_parameter_fraction(
        MODELS[config.workload.model]
    )
    layer_timings: list[LayerTiming] = []
    token_timings: list[TokenTiming] = []
    next_collective_index = 0
    start_state = _single(
        config.workload.start_state_values, "workload.start_state"
    )
    warmup_tokens = (
        1
        if start_state == "steady_state"
        and run_mode in {"decode_only", "prefill_decode"}
        else 0
    )
    total_decode_steps = decode_tokens + warmup_tokens

    def execute_step(
        *,
        phase: str,
        plan_token_index: int,
        trace_token_index: int,
        measured: bool,
    ):
        nonlocal next_collective_index
        plans = build_layer_plans(
            config.workload.model,
            config.workload.tp,
            config.workload.seq_len,
            batch,
            phase=phase,
            decode_tokens=max(total_decode_steps, 1),
            token_index=(plan_token_index if phase == "decode" else 0),
        )
        kv_start = math.fsum(
            plan.lanes[0].historical_kv_bytes for plan in plans
        )
        kv_increment = math.fsum(
            plan.lanes[0].new_kv_bytes for plan in plans
        )
        token_started = env.now
        prefetch_events: dict[int, simpy.Process] = {}
        prefetch_launches: dict[int, float] = {}
        writebacks: list[tuple[int, simpy.Process]] = []
        timing_start = len(layer_timings)

        def launch_prefetch(layer_index: int) -> None:
            if layer_index >= len(plans) or layer_index in prefetch_events:
                return
            prefetch_launches[layer_index] = env.now
            prefetch_events[layer_index] = env.process(
                _prefetch_layer(
                    env,
                    plans[layer_index],
                    config,
                    rank_resources,
                    pools,
                    residency.effective_policy,
                )
            )

        launch_prefetch(0)
        previous_layer_end = token_started
        for layer_index, plan in enumerate(plans):
            yield prefetch_events[layer_index]
            readiness = prefetch_events[layer_index].value
            lane_ready = readiness.lane_ready_s
            layer_ready = max(lane_ready)
            compute_start = env.now
            launch_prefetch(layer_index + 1)

            attention_amounts = tuple(
                lane.weight_bytes * attention_fraction
                + lane.historical_kv_bytes
                for lane in plan.lanes
            )
            yield env.process(
                _local_access(
                    env,
                    rank_resources,
                    attention_amounts,
                    (
                        TrafficClass.UHBM_GPU_READ
                        if config.architectures[0].kind == "b2_uhbm"
                        else TrafficClass.HBM_2_5D_GPU_READ
                    ),
                )
            )
            attention_compute_start = env.now
            yield env.process(
                _compute_all_ranks(
                    env, collective_resources, plan.attention_compute_flops
                )
            )
            attention_compute_end = env.now
            writeback = env.process(
                _writeback_layer(
                    env, plan, config, rank_resources, pools
                )
            )
            writebacks.append((len(layer_timings), writeback))
            collective_base = next_collective_index
            next_collective_index += 2
            yield env.process(
                _collective(
                    env,
                    coordinator,
                    collective_base,
                    plan.attention_collective_bytes,
                    config.execution.r,
                    config.workload.tp,
                )
            )
            attention_collective_end = env.now

            mlp_amounts = tuple(
                lane.weight_bytes * (1.0 - attention_fraction)
                for lane in plan.lanes
            )
            yield env.process(
                _local_access(
                    env,
                    rank_resources,
                    mlp_amounts,
                    (
                        TrafficClass.UHBM_GPU_READ
                        if config.architectures[0].kind == "b2_uhbm"
                        else TrafficClass.HBM_2_5D_GPU_READ
                    ),
                )
            )
            mlp_compute_start = env.now
            yield env.process(
                _compute_all_ranks(
                    env, collective_resources, plan.mlp_compute_flops
                )
            )
            mlp_compute_end = env.now
            yield env.process(
                _collective(
                    env,
                    coordinator,
                    collective_base + 1,
                    plan.mlp_collective_bytes,
                    config.execution.r,
                    config.workload.tp,
                )
            )
            compute_end = env.now
            slack = previous_layer_end - layer_ready
            layer_timings.append(
                LayerTiming(
                    token_index=trace_token_index,
                    layer_index=layer_index,
                    prefetch_launch_s=prefetch_launches[layer_index],
                    lane_ready_s=lane_ready,
                    weight_lane_ready_s=readiness.weight_lane_ready_s,
                    kv_lane_ready_s=readiness.kv_lane_ready_s,
                    layer_ready_s=layer_ready,
                    previous_layer_end_s=previous_layer_end,
                    compute_start_s=compute_start,
                    attention_compute_start_s=attention_compute_start,
                    attention_compute_end_s=attention_compute_end,
                    attention_collective_end_s=attention_collective_end,
                    mlp_compute_start_s=mlp_compute_start,
                    mlp_compute_end_s=mlp_compute_end,
                    mlp_collective_end_s=compute_end,
                    compute_end_s=compute_end,
                    prefetch_slack_s=slack,
                    prefetch_stall_s=max(0.0, -slack),
                    writeback_end_s=None,
                )
            )
            previous_layer_end = compute_end
        if writebacks:
            yield env.all_of([event for _, event in writebacks])
            for timing_index, event in writebacks:
                layer_timings[timing_index] = replace(
                    layer_timings[timing_index],
                    writeback_end_s=event.value,
                )
        token_ended = env.now
        token_timings.append(
            TokenTiming(
                token_index=trace_token_index,
                phase=phase,
                started_s=token_started,
                ended_s=token_ended,
                latency_s=token_ended - token_started,
                measured=measured,
                kv_resident_bytes_per_lane_start=kv_start,
                kv_resident_bytes_per_lane_end=kv_start + kv_increment,
            )
        )
        assert len(layer_timings) - timing_start == len(plans)

    def execute_run():
        if run_mode in {"prefill_only", "prefill_decode"}:
            yield env.process(
                execute_step(
                    phase="prefill",
                    plan_token_index=0,
                    trace_token_index=0,
                    measured=True,
                )
            )
        if run_mode in {"decode_only", "prefill_decode"}:
            for decode_index in range(total_decode_steps):
                is_warmup = decode_index < warmup_tokens
                measured_index = decode_index - warmup_tokens
                yield env.process(
                    execute_step(
                        phase="decode",
                        plan_token_index=decode_index,
                        trace_token_index=(-1 if is_warmup else measured_index),
                        measured=not is_warmup,
                    )
                )

    run_process = env.process(execute_run())
    env.run()
    if not run_process.triggered:
        raise RuntimeError("prefetch run did not complete")

    measured_decode = tuple(
        timing.latency_s
        for timing in token_timings
        if timing.phase == "decode" and timing.measured
    )
    prefill_timing = next(
        (timing for timing in token_timings if timing.phase == "prefill"),
        None,
    )
    first_measured_decode = next(
        (
            timing
            for timing in token_timings
            if timing.phase == "decode" and timing.measured
        ),
        None,
    )
    ttft = (
        prefill_timing.latency_s
        if prefill_timing is not None
        else (
            first_measured_decode.latency_s
            if first_measured_decode is not None
            else None
        )
    )
    all_resources = tuple(
        resource
        for layer in rank_resources
        for resource in layer.all_resources()
    ) + tuple(collective_resources.h2h_link.values()) + tuple(
        collective_resources.gpu_direct_link.values()
    ) + tuple(collective_resources.gpu_compute.values())
    return PrefetchSimulationResult(
        feasible=True,
        infeasibility_reason=None,
        first_infeasible_token=None,
        residency=residency,
        elapsed_s=env.now,
        ttft_s=ttft,
        first_token_latency_s=(
            first_measured_decode.latency_s
            if first_measured_decode is not None
            else None
        ),
        token_latencies_s=measured_decode,
        tpot_s=(math.fsum(measured_decode) / len(measured_decode) if measured_decode else None),
        path_bytes=_path_bytes(all_resources),
        trace=(
            PrefetchTrace(tuple(layer_timings), tuple(token_timings))
            if detailed_trace
            else None
        ),
        resources=all_resources,
    )
