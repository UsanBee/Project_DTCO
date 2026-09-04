"""Stage 2 SimPy discrete-event simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import simpy

from uhbm_simulator.resources import RateResource, build_resource_layer
from uhbm_simulator.workload import TrafficSignature, TrafficStreams, split_traffic


# TODO(태현): 근거 없음 -- 아직 제작되지 않은 uHBM 패키지의 thermal 시뮬레이션 결과가
# 없어 공개 문헌으로 이 배수를 특정할 수 없었음. spec 5장 의도대로 "측정값"이 아니라
# sensitivity 시나리오 파라미터로 유지 (0.85/0.65는 여전히 임의값).
THERMAL_PERFORMANCE_FACTORS = {
    "no_throttling": 1.0,
    "mild_throttling": 0.85,
    "severe_throttling": 0.65,
}


class LocalHBMCapacityError(ValueError):
    """Raised when one local HBM stack cannot hold its assigned working set."""

    def __init__(self, stack_index: int, message: str) -> None:
        super().__init__(message)
        self.stack_index = stack_index


@dataclass(frozen=True)
class SimulationParameters:
    tp: int
    r: float
    r_remote: float
    comm_mode: str
    thermal_scenario: str
    local_hbm_stack_count: int
    local_hbm_stack_bw_bps: float
    local_hbm_stack_capacity_bytes: float
    local_hbm_mapping_policy: str
    local_hbm_mapping_weights: tuple[float, ...] | None
    side_hbm_bw_bps: float
    side_hbm_latency_s: float
    h2h_bw_bps: float
    h2h_latency_s: float
    gpu_direct_bw_bps: float
    gpu_direct_latency_s: float
    gpu_peak_flops: float
    base_die_latency_s: float
    include_materialized_endpoints: bool = True
    allow_overlap: bool = True


@dataclass(frozen=True)
class ResourceMetric:
    request_count: int
    transferred_amount: float
    total_queue_time_s: float
    total_service_time_s: float
    total_latency_time_s: float
    first_service_start_s: float | None
    utilization: float


@dataclass(frozen=True)
class SimulationResult:
    ttft_s: float
    tpot_s: float
    communication_time_s: float
    overlap_s: float
    resource_metrics: dict[str, ResourceMetric]


def _use(resource: RateResource, amount: float):
    if amount > 0.0:
        yield resource.env.process(resource.use(amount))


def _local_hbm_fractions(parameters: SimulationParameters) -> tuple[float, ...]:
    count = parameters.local_hbm_stack_count
    if count < 1:
        raise ValueError("local_hbm_stack_count must be positive")
    if parameters.local_hbm_mapping_policy == "uniform":
        return (1.0 / count,) * count
    if parameters.local_hbm_mapping_policy != "weighted":
        raise ValueError("local_hbm_mapping_policy must be uniform or weighted")

    weights = parameters.local_hbm_mapping_weights
    if (
        weights is None
        or len(weights) != count
        or any(not math.isfinite(weight) or weight < 0.0 for weight in weights)
        or max(weights, default=0.0) <= 0.0
    ):
        raise ValueError(
            "weighted local_hbm_mapping_weights must match stack count, be finite "
            "and nonnegative, and contain at least one positive value"
        )
    scale = max(weights)
    scaled_weights = tuple(weight / scale for weight in weights)
    scaled_total = sum(scaled_weights)
    return tuple(weight / scaled_total for weight in scaled_weights)


def _use_local_hbm(
    env: simpy.Environment,
    resources,
    tile: int,
    amount: float,
    parameters: SimulationParameters,
):
    events = [
        env.process(resources.local_hbm[(tile, stack)].use(amount * fraction))
        for stack, fraction in enumerate(_local_hbm_fractions(parameters))
        if amount > 0.0 and fraction > 0.0
    ]
    if events:
        yield env.all_of(events)


def _memory_compute_process(
    env: simpy.Environment,
    tile: int,
    streams: TrafficStreams,
    signature: TrafficSignature,
    parameters: SimulationParameters,
    resources,
):
    memory_events = []
    if streams.local_hbm_bytes > 0:
        memory_events.append(
            env.process(
                _use_local_hbm(
                    env, resources, tile, streams.local_hbm_bytes, parameters
                )
            )
        )
    if streams.side_hbm_bytes > 0:
        side_amount = streams.side_hbm_bytes / streams.n_side_requests

        def side_requests():
            for _ in range(streams.n_side_requests):
                yield env.process(resources.side_hbm[tile].use(side_amount))

        memory_events.append(
            env.process(side_requests())
        )
    if memory_events:
        yield env.all_of(memory_events)
    if signature.compute_flops > 0:
        yield env.process(resources.gpu_compute[tile].use(signature.compute_flops))


def _materialized_h2h_message(
    env: simpy.Environment,
    source: int,
    destination: int,
    amount: float,
    parameters: SimulationParameters,
    resources,
):
    if parameters.include_materialized_endpoints:
        yield env.process(
            _use_local_hbm(env, resources, source, amount, parameters)
        )
    yield env.process(resources.h2h_link[(source, destination)].use(amount))
    if parameters.base_die_latency_s > 0:
        yield env.timeout(parameters.base_die_latency_s)
    if parameters.include_materialized_endpoints:
        yield env.process(
            _use_local_hbm(env, resources, destination, amount, parameters)
        )
        yield env.process(
            _use_local_hbm(env, resources, destination, amount, parameters)
        )


def _cut_through_h2h_message(
    env: simpy.Environment,
    source: int,
    destination: int,
    amount: float,
    parameters: SimulationParameters,
    resources,
):
    yield env.process(_use_local_hbm(env, resources, source, amount, parameters))
    yield env.process(resources.h2h_link[(source, destination)].use(amount))
    if parameters.base_die_latency_s > 0:
        yield env.timeout(parameters.base_die_latency_s)


def _communication_process(
    env: simpy.Environment,
    tile: int,
    streams: TrafficStreams,
    parameters: SimulationParameters,
    resources,
    intervals: dict[int, list[tuple[float, float]]],
):
    started = env.now
    if streams.n_messages <= 0 or parameters.tp <= 1:
        intervals[tile] = []
        return

    destination = (tile + 1) % parameters.tp
    h2h_per_message = streams.h2h_bytes / streams.n_messages
    direct_per_message = streams.gpu_direct_bytes / streams.n_messages
    for _ in range(streams.n_messages):
        yield env.process(
            _communication_message(
                env,
                tile,
                destination,
                h2h_per_message,
                direct_per_message,
                parameters,
                resources,
            )
        )
    intervals[tile] = [(started, env.now)]


def _communication_message(
    env: simpy.Environment,
    source: int,
    destination: int,
    h2h_amount: float,
    direct_amount: float,
    parameters: SimulationParameters,
    resources,
):
    events = []
    if h2h_amount > 0:
        h2h_process = (
            _materialized_h2h_message
            if parameters.comm_mode == "materialized"
            else _cut_through_h2h_message
        )
        events.append(
            env.process(
                h2h_process(
                    env,
                    source,
                    destination,
                    h2h_amount,
                    parameters,
                    resources,
                )
            )
        )
    if direct_amount > 0:
        events.append(
            env.process(
                resources.gpu_direct_link[(source, destination)].use(direct_amount)
            )
        )
    if events:
        yield env.all_of(events)


def _causal_tile_process(
    env: simpy.Environment,
    tile: int,
    streams: TrafficStreams,
    signature: TrafficSignature,
    parameters: SimulationParameters,
    resources,
    communication_intervals: dict[int, list[tuple[float, float]]],
):
    stage_count = max(streams.n_messages, streams.n_side_requests, 1)
    local_per_stage = streams.local_hbm_bytes / stage_count
    compute_per_stage = signature.compute_flops / stage_count
    side_per_request = (
        streams.side_hbm_bytes / streams.n_side_requests
        if streams.n_side_requests
        else 0.0
    )
    h2h_per_message = (
        streams.h2h_bytes / streams.n_messages if streams.n_messages else 0.0
    )
    direct_per_message = (
        streams.gpu_direct_bytes / streams.n_messages
        if streams.n_messages
        else 0.0
    )
    destination = (tile + 1) % parameters.tp if parameters.tp > 1 else tile
    communication_events = []

    def timed_message():
        started = env.now
        yield env.process(
            _communication_message(
                env,
                tile,
                destination,
                h2h_per_message,
                direct_per_message,
                parameters,
                resources,
            )
        )
        communication_intervals.setdefault(tile, []).append((started, env.now))

    for stage in range(stage_count):
        memory_events = []
        if local_per_stage > 0:
            memory_events.append(
                env.process(
                    _use_local_hbm(
                        env, resources, tile, local_per_stage, parameters
                    )
                )
            )
        if stage < streams.n_side_requests:
            memory_events.append(
                env.process(resources.side_hbm[tile].use(side_per_request))
            )
        if memory_events:
            yield env.all_of(memory_events)
        if compute_per_stage > 0:
            yield env.process(resources.gpu_compute[tile].use(compute_per_stage))

        if stage < streams.n_messages and parameters.tp > 1:
            communication_events.append(env.process(timed_message()))

    if communication_events:
        yield env.all_of(communication_events)
    else:
        communication_intervals[tile] = []


def _interval_union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _snapshot_metrics(resources, elapsed_s: float) -> dict[str, ResourceMetric]:
    metrics = {}
    for resource in resources.all_resources():
        stats = resource.stats
        metrics[resource.name] = ResourceMetric(
            request_count=stats.request_count,
            transferred_amount=stats.transferred_amount,
            total_queue_time_s=stats.total_queue_time_s,
            total_service_time_s=stats.total_service_time_s,
            total_latency_time_s=stats.total_latency_time_s,
            first_service_start_s=stats.first_service_start_s,
            utilization=stats.utilization(elapsed_s),
        )
    return metrics


def run_simulation(
    signature: TrafficSignature,
    parameters: SimulationParameters,
) -> SimulationResult:
    """Run one representative Stage 2 workload step across a TP ring."""
    if parameters.comm_mode not in {"materialized", "cut_through"}:
        raise ValueError("comm_mode must be materialized or cut_through")
    if parameters.thermal_scenario not in THERMAL_PERFORMANCE_FACTORS:
        raise ValueError("unknown thermal_scenario")
    if parameters.tp < 1:
        raise ValueError("tp must be positive")
    if (
        not math.isfinite(parameters.local_hbm_stack_bw_bps)
        or parameters.local_hbm_stack_bw_bps <= 0.0
    ):
        raise ValueError("local_hbm stack bandwidth must be positive and finite")
    capacity_is_fixture_infinity = (
        parameters.local_hbm_stack_capacity_bytes == math.inf
        and not parameters.include_materialized_endpoints
    )
    if (
        not capacity_is_fixture_infinity
        and (
            not math.isfinite(parameters.local_hbm_stack_capacity_bytes)
            or parameters.local_hbm_stack_capacity_bytes <= 0.0
        )
    ):
        raise ValueError("local_hbm stack capacity must be positive and finite")

    streams = split_traffic(signature, parameters.r_remote, parameters.r)
    local_hbm_fractions = _local_hbm_fractions(parameters)
    if streams.side_hbm_bytes > 0.0 and streams.n_side_requests <= 0:
        raise ValueError("nonzero side-HBM traffic requires a positive request count")
    if (
        streams.h2h_bytes + streams.gpu_direct_bytes > 0.0
        and streams.n_messages <= 0
    ):
        raise ValueError("nonzero collective traffic requires a positive request count")
    materialized_buffer = (
        streams.h2h_bytes / streams.n_messages
        if (
            parameters.comm_mode == "materialized"
            and parameters.include_materialized_endpoints
            and streams.h2h_bytes > 0.0
        )
        else 0.0
    )
    required_local_capacity = streams.local_hbm_bytes + materialized_buffer
    for stack, fraction in enumerate(local_hbm_fractions):
        if (
            required_local_capacity * fraction
            > parameters.local_hbm_stack_capacity_bytes
        ):
            raise LocalHBMCapacityError(
                stack,
                f"local_hbm stack {stack} working set and materialized receive "
                "buffer exceed per-stack capacity; increase r_remote/r or capacity"
            )

    env = simpy.Environment()
    effective_gpu_flops = (
        parameters.gpu_peak_flops
        * THERMAL_PERFORMANCE_FACTORS[parameters.thermal_scenario]
    )
    resources = build_resource_layer(
        env=env,
        tp=parameters.tp,
        r=parameters.r,
        local_hbm_stack_count=parameters.local_hbm_stack_count,
        local_hbm_stack_bw_bps=parameters.local_hbm_stack_bw_bps,
        local_hbm_stack_capacity_bytes=parameters.local_hbm_stack_capacity_bytes,
        side_hbm_bw_bps=parameters.side_hbm_bw_bps,
        side_hbm_latency_s=parameters.side_hbm_latency_s,
        h2h_bw_bps=parameters.h2h_bw_bps,
        h2h_latency_s=parameters.h2h_latency_s,
        gpu_direct_bw_bps=parameters.gpu_direct_bw_bps,
        gpu_direct_latency_s=parameters.gpu_direct_latency_s,
        gpu_compute_flops=effective_gpu_flops,
    )

    communication_intervals: dict[int, list[tuple[float, float]]] = {}
    completion_times: dict[int, float] = {}

    def tile_process(tile: int):
        if parameters.allow_overlap:
            yield env.process(
                _causal_tile_process(
                    env,
                    tile,
                    streams,
                    signature,
                    parameters,
                    resources,
                    communication_intervals,
                )
            )
        else:
            yield env.process(
                _memory_compute_process(
                    env, tile, streams, signature, parameters, resources
                )
            )
            yield env.process(
                _communication_process(
                    env,
                    tile,
                    streams,
                    parameters,
                    resources,
                    communication_intervals,
                )
            )
        completion_times[tile] = env.now

    for tile in range(parameters.tp):
        env.process(tile_process(tile))
    env.run()

    elapsed_s = max(completion_times.values(), default=0.0)
    communication_time_s = max(
        (
            _interval_union_duration(intervals)
            for intervals in communication_intervals.values()
        ),
        default=0.0,
    )
    if parameters.allow_overlap:
        sequential = run_simulation(
            signature, replace(parameters, allow_overlap=False)
        )
        overlap_s = max(0.0, sequential.ttft_s - elapsed_s)
    else:
        overlap_s = 0.0
    return SimulationResult(
        ttft_s=elapsed_s,
        tpot_s=elapsed_s / signature.token_count,
        communication_time_s=communication_time_s,
        overlap_s=overlap_s,
        resource_metrics=_snapshot_metrics(resources, elapsed_s),
    )
