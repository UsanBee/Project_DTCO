"""Stage 2 SimPy discrete-event simulator."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import simpy

from stage1_model import ring_allreduce_round_count
from uhbm_simulator.config import ChannelsConfig, h2h_energy_pj_per_bit
from uhbm_simulator.resources import RateResource, build_resource_layer
from uhbm_simulator.workload import (
    SideHBMCapacityError,
    StageTrafficStreams,
    TrafficSignature,
    TrafficStreams,
    _split_evenly,
    split_traffic,
)


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
    # Placement fraction applied independently to local-memory transfer traffic
    # and resident footprint; accesses are assumed to follow that placement.
    # Local HBM is capacity-constrained. Side HBM remains unconstrained in the
    # default fixed-fraction mode; capacity-overflow placement can optionally
    # constrain its resident footprint through side_hbm_capacity_bytes.
    # `r_remote` only takes effect when local_hbm_placement_mode ==
    # "fixed_fraction". In "capacity_overflow" mode this input is ignored
    # entirely; the effective local/remote split is an output of
    # capacity-driven placement, exposed as TrafficStreams.effective_r_remote.
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
    # Preserve the existing zero-latency approximation for nearby uHBM local
    # stacks while allowing future conventional 2.5D baselines to specify a
    # meaningful local-memory access latency.
    local_hbm_latency_s: float = 0.0
    include_materialized_endpoints: bool = True
    allow_overlap: bool = True
    # Production default: dependent compute must wait for the preceding
    # collective, while its memory prefetch may overlap that communication.
    # Set False explicitly only to reproduce the legacy loose DAG for
    # diagnostics/comparison; the loose process remains intentionally available.
    # For Megatron-style tensor parallelism, strict mode enforces the required
    # intermediate AllReduce dependency between sub-layers. Representative
    # timeline audit confirms that the next compute stage does not begin before
    # the previous collective completes (see uhbm-simulator scope-check log,
    # 2026-09-08).
    strict_collective_dependency: bool = True
    local_hbm_placement_mode: str = "fixed_fraction"
    # Per-GPU side-memory resident capacity used only by capacity_overflow;
    # None retains the capacity-unconstrained pooled-memory approximation.
    side_hbm_capacity_bytes: float | None = None
    # Event-driven bounding cases for local-vs-side memory concurrency:
    # full_overlap is the optimistic performance bound (latency lower bound),
    # while serialized is the conservative no-concurrency bound. Neither is
    # asserted to be the validated behavior of future uHBM hardware.
    local_side_memory_concurrency: str = "full_overlap"
    # shared_h2h is the production uHBM interpretation: remote-memory payloads
    # use the same directed H2H ring edges as TP collectives. independent keeps
    # the former no-H2H side-memory path as a research comparison case;
    # pillar_bypass uses a finite, physically separate D-3 transport resource.
    side_hbm_path: str = "shared_h2h"
    # Ownership is independent of transport so same-owner controlled
    # comparisons isolate contention removal from locality changes.
    side_hbm_owner_scope: str = "all_remote"
    # For adjacent ownership, entry t lists physical neighbors of requester t.
    # No automatic topology is inferred in Stage 2.
    physical_adjacency: tuple[tuple[int, ...], ...] | None = None
    # Nominal pillar transport anchors, pending Stage 3 HFSS/ADS calibration.
    pillar_link_bw_bps: float | None = None
    pillar_link_latency_s: float | None = None


@dataclass(frozen=True)
class ResourceMetric:
    request_count: int
    transferred_amount: float
    total_queue_time_s: float
    total_service_time_s: float
    total_latency_time_s: float
    first_service_start_s: float | None
    utilization: float
    by_traffic_class: dict[str, "ResourceMetric"] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResult:
    elapsed_s: float
    ttft_s: float
    tpot_s: float
    communication_time_s: float
    # Counterfactual saved time, not a direct measurement of resource-level
    # overlap: overlap_s = T_sequential - T_overlap_enabled, where T_sequential
    # re-runs the same signature and parameters with allow_overlap=False. It
    # does not identify which resources overlapped or by how much.
    overlap_s: float
    # Actual system-wide bytes placed on physical H2H edges, separated by
    # traffic source. Their sum equals the h2h_link resource byte total.
    collective_h2h_edge_bytes: float
    side_hbm_read_h2h_edge_bytes: float
    pillar_side_read_edge_bytes: float
    resource_metrics: dict[str, ResourceMetric]


@dataclass
class H2HTrafficCounters:
    """Accumulate completed edge-byte transfers without utilization inference."""

    collective_edge_amounts: list[float] = field(default_factory=list)
    side_read_edge_amounts: list[float] = field(default_factory=list)
    pillar_side_read_edge_amounts: list[float] = field(default_factory=list)

    @property
    def collective_h2h_edge_bytes(self) -> float:
        return math.fsum(self.collective_edge_amounts)

    @property
    def side_hbm_read_h2h_edge_bytes(self) -> float:
        return math.fsum(self.side_read_edge_amounts)

    @property
    def pillar_side_read_edge_bytes(self) -> float:
        return math.fsum(self.pillar_side_read_edge_amounts)


@dataclass(frozen=True)
class ChannelEnergy:
    local_hbm_energy_j: float
    local_hbm_idle_energy_j: float
    side_hbm_energy_j: float
    side_hbm_link_energy_j: float
    h2h_energy_j: float
    pillar_link_energy_j: float
    gpu_direct_energy_j: float
    total_energy_j: float


def _transferred_bytes(
    metrics: dict[str, ResourceMetric], prefix: str
) -> float:
    return sum(
        metric.transferred_amount
        for name, metric in metrics.items()
        if name.startswith(prefix)
    )


def _energy_j(transferred_bytes: float, energy_pj_per_bit: float) -> float:
    return transferred_bytes * 8.0 * energy_pj_per_bit * 1e-12


def calculate_channel_energy(
    metrics: dict[str, ResourceMetric],
    channels: ChannelsConfig,
    lane_rate_gbps: float,
    local_hbm_stack_count: int,
    elapsed_s: float,
    tp: int,
    side_hbm_path: str = "independent",
    pillar_link_energy_pj_per_bit: float | None = None,
) -> ChannelEnergy:
    """Calculate system-wide transfer and powered-stack background energy."""
    local = _energy_j(
        _transferred_bytes(metrics, "local_hbm"),
        channels.local_hbm.stack.energy_pj_per_bit,
    )
    local_idle = (
        tp
        * local_hbm_stack_count
        * channels.local_hbm.stack.hbm_idle_power_w
        * elapsed_s
    )
    side_bytes = _transferred_bytes(metrics, "side_hbm")
    # One remote-memory datapath charge and one physically distinct crossing
    # charge use the same exact bytes; neither is inferred from utilization.
    side = _energy_j(
        side_bytes,
        channels.side_hbm.energy_pj_per_bit,
    )
    if side_hbm_path not in {"shared_h2h", "independent", "pillar_bypass"}:
        raise ValueError(
            "side_hbm_path must be shared_h2h, independent, or pillar_bypass"
        )
    # independent represents the former separately charged crossing fabric.
    # shared_h2h bytes are already counted on h2h_link resources, so charging
    # the side-link term again would double-count the same physical transfer.
    side_link = (
        _energy_j(side_bytes, channels.side_hbm.link_energy_pj_per_bit)
        if side_hbm_path == "independent"
        else 0.0
    )
    h2h = _energy_j(
        _transferred_bytes(metrics, "h2h_link"),
        h2h_energy_pj_per_bit(lane_rate_gbps),
    )
    if side_hbm_path == "pillar_bypass":
        if (
            pillar_link_energy_pj_per_bit is None
            or not math.isfinite(pillar_link_energy_pj_per_bit)
            or pillar_link_energy_pj_per_bit <= 0.0
        ):
            raise ValueError(
                "positive pillar_link_energy_pj_per_bit is required for pillar_bypass"
            )
        pillar = _energy_j(
            _transferred_bytes(metrics, "pillar_link"),
            pillar_link_energy_pj_per_bit,
        )
    else:
        pillar = 0.0
    direct = _energy_j(
        _transferred_bytes(metrics, "gpu_direct_link"),
        channels.gpu_direct.energy_pj_per_bit,
    )
    return ChannelEnergy(
        local_hbm_energy_j=local,
        local_hbm_idle_energy_j=local_idle,
        side_hbm_energy_j=side,
        side_hbm_link_energy_j=side_link,
        h2h_energy_j=h2h,
        pillar_link_energy_j=pillar,
        gpu_direct_energy_j=direct,
        total_energy_j=(
            local + local_idle + side + side_link + h2h + pillar + direct
        ),
    )


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


def _validate_physical_adjacency(
    physical_adjacency: tuple[tuple[int, ...], ...] | None,
    tp: int,
) -> None:
    """Validate an explicitly supplied undirected physical-neighbor graph."""
    if physical_adjacency is None:
        raise ValueError("physical_adjacency is required for adjacent ownership")
    if len(physical_adjacency) != tp:
        raise ValueError("physical_adjacency must have exactly one entry per TP tile")
    for requester, owners in enumerate(physical_adjacency):
        if not owners:
            raise ValueError("each tile must have at least one adjacent owner")
        if len(set(owners)) != len(owners):
            raise ValueError("physical_adjacency neighbor entries must be unique")
        for owner in owners:
            if isinstance(owner, bool) or not isinstance(owner, int):
                raise ValueError("physical_adjacency tile indices must be integers")
            if owner < 0 or owner >= tp:
                raise ValueError("physical_adjacency tile index is out of range")
            if owner == requester:
                raise ValueError("physical_adjacency cannot contain a self edge")
    for requester, owners in enumerate(physical_adjacency):
        for owner in owners:
            if requester not in physical_adjacency[owner]:
                raise ValueError("physical_adjacency must be symmetric")


def _validate_side_hbm_architecture(parameters: SimulationParameters) -> None:
    if parameters.side_hbm_path not in {
        "shared_h2h",
        "independent",
        "pillar_bypass",
    }:
        raise ValueError(
            "side_hbm_path must be shared_h2h, independent, or pillar_bypass"
        )
    if parameters.side_hbm_owner_scope not in {"all_remote", "adjacent"}:
        raise ValueError("side_hbm_owner_scope must be all_remote or adjacent")
    if parameters.side_hbm_path == "pillar_bypass" and (
        parameters.side_hbm_owner_scope != "adjacent"
    ):
        raise ValueError("pillar_bypass requires side_hbm_owner_scope=adjacent")
    if parameters.tp <= 1:
        return
    if parameters.side_hbm_owner_scope == "adjacent":
        _validate_physical_adjacency(parameters.physical_adjacency, parameters.tp)
    if parameters.side_hbm_path == "pillar_bypass":
        if (
            parameters.pillar_link_bw_bps is None
            or not math.isfinite(parameters.pillar_link_bw_bps)
            or parameters.pillar_link_bw_bps <= 0.0
        ):
            raise ValueError(
                "positive pillar_link_bw_bps is required for pillar_bypass"
            )
        if (
            parameters.pillar_link_latency_s is None
            or not math.isfinite(parameters.pillar_link_latency_s)
            or parameters.pillar_link_latency_s < 0.0
        ):
            raise ValueError(
                "nonnegative pillar_link_latency_s is required for pillar_bypass"
            )


def _side_hbm_owner(
    requester: int,
    request_index: int,
    parameters: SimulationParameters,
) -> int:
    if parameters.side_hbm_owner_scope == "all_remote":
        return (
            requester + 1 + request_index % (parameters.tp - 1)
        ) % parameters.tp
    owners = parameters.physical_adjacency[requester]
    return owners[request_index % len(owners)]


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


def _side_hbm_read_process(
    env: simpy.Environment,
    requester: int,
    amount: float,
    request_index: int,
    parameters: SimulationParameters,
    resources,
    counters: H2HTrafficCounters,
):
    """Serve a side-HBM read and return data over the selected transport.

    Request/control bytes are omitted in this MVP. Only the dominant read data
    payload is transported, making shared_h2h an optimistic control-overhead
    approximation. Deterministic pooled ownership avoids page-level modeling.
    """
    if amount <= 0.0:
        return
    if parameters.tp == 1:
        yield env.process(resources.side_hbm[requester].use(amount))
        return
    if (
        parameters.side_hbm_path == "independent"
        and parameters.side_hbm_owner_scope == "all_remote"
    ):
        # Preserve the legacy independent path bit-for-bit: it was modeled as
        # requester-indexed pooled service with no explicit remote owner.
        yield env.process(resources.side_hbm[requester].use(amount))
        return

    owner = _side_hbm_owner(requester, request_index, parameters)
    yield env.process(resources.side_hbm[owner].use(amount))
    if parameters.side_hbm_path == "independent":
        return
    if parameters.side_hbm_path == "pillar_bypass":
        # Remote-read payload direction is owner -> requester. Request/control
        # bytes are omitted in this v1 optimistic approximation. Opposite
        # directed pillar resources are independent, modeling full duplex.
        yield env.process(resources.pillar_link[(owner, requester)].use(amount))
        counters.pillar_side_read_edge_amounts.append(amount)
        return
    # The modeled traffic is a remote read: the small requester->owner control
    # request is omitted, and the large data payload travels owner->requester.
    for edge in _directed_ring_path(owner, requester, parameters.tp):
        yield env.process(
            _h2h_edge_transfer(
                env,
                edge,
                amount,
                parameters,
                resources,
                counters.side_read_edge_amounts,
            )
        )


def _serialized_local_side_memory_process(
    env: simpy.Environment,
    tile: int,
    local_amount: float,
    side_amount_per_request: float,
    side_request_count: int,
    parameters: SimulationParameters,
    resources,
    counters: H2HTrafficCounters,
    side_request_start_index: int = 0,
):
    """Run the conservative local-then-side memory bounding schedule."""
    if local_amount > 0.0:
        yield env.process(
            _use_local_hbm(env, resources, tile, local_amount, parameters)
        )
    for offset in range(side_request_count):
        yield env.process(
            _side_hbm_read_process(
                env,
                tile,
                side_amount_per_request,
                side_request_start_index + offset,
                parameters,
                resources,
                counters,
            )
        )


def _memory_compute_process(
    env: simpy.Environment,
    tile: int,
    streams: TrafficStreams,
    signature: TrafficSignature,
    parameters: SimulationParameters,
    resources,
    counters: H2HTrafficCounters,
):
    if parameters.local_side_memory_concurrency == "full_overlap":
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
                for request_index in range(streams.n_side_requests):
                    yield env.process(
                        _side_hbm_read_process(
                            env,
                            tile,
                            side_amount,
                            request_index,
                            parameters,
                            resources,
                            counters,
                        )
                    )

            memory_events.append(
                env.process(side_requests())
            )
        if memory_events:
            yield env.all_of(memory_events)
    else:
        side_amount = (
            streams.side_hbm_bytes / streams.n_side_requests
            if streams.n_side_requests
            else 0.0
        )
        yield env.process(
            _serialized_local_side_memory_process(
                env,
                tile,
                streams.local_hbm_bytes,
                side_amount,
                streams.n_side_requests if streams.side_hbm_bytes > 0.0 else 0,
                parameters,
                resources,
                counters,
            )
        )
    if signature.compute_flops > 0:
        yield env.process(resources.gpu_compute[tile].use(signature.compute_flops))


def _semantic_stage_memory_process(
    env: simpy.Environment,
    tile: int,
    stage: StageTrafficStreams,
    parameters: SimulationParameters,
    resources,
    counters: H2HTrafficCounters,
    side_request_index: int | None,
):
    """Run one semantic stage's placement-aware memory accesses."""
    if parameters.local_side_memory_concurrency == "full_overlap":
        memory_events = []
        if stage.local_hbm_bytes > 0.0:
            memory_events.append(
                env.process(
                    _use_local_hbm(
                        env,
                        resources,
                        tile,
                        stage.local_hbm_bytes,
                        parameters,
                    )
                )
            )
        if stage.side_hbm_bytes > 0.0:
            memory_events.append(
                env.process(
                    _side_hbm_read_process(
                        env,
                        tile,
                        stage.side_hbm_bytes,
                        side_request_index,
                        parameters,
                        resources,
                        counters,
                    )
                )
            )
        if memory_events:
            yield env.all_of(memory_events)
    else:
        yield env.process(
            _serialized_local_side_memory_process(
                env,
                tile,
                stage.local_hbm_bytes,
                stage.side_hbm_bytes,
                1 if stage.side_hbm_bytes > 0.0 else 0,
                parameters,
                resources,
                counters,
                side_request_index or 0,
            )
        )


def _split_collective_round_bytes(amount: float, tp: int) -> tuple[float, ...]:
    """Split one aggregate per-rank H2H volume over explicit ring rounds."""
    if amount < 0.0:
        raise ValueError("collective bytes cannot be negative")
    rounds = ring_allreduce_round_count(tp)
    if rounds == 0:
        if amount > 0.0:
            raise ValueError("TP=1 cannot carry H2H collective bytes")
        return ()
    return _split_evenly(amount, rounds)


def _directed_ring_path(
    source: int,
    destination: int,
    tp: int,
) -> tuple[tuple[int, int], ...]:
    """Return the explicit forward-edge path on the unidirectional TP ring."""
    if tp < 1:
        raise ValueError("tp must be positive")
    if not 0 <= source < tp or not 0 <= destination < tp:
        raise ValueError("ring endpoints must be valid tile indices")
    edges = []
    current = source
    while current != destination:
        successor = (current + 1) % tp
        edges.append((current, successor))
        current = successor
    return tuple(edges)


def _h2h_edge_transfer(
    env: simpy.Environment,
    edge: tuple[int, int],
    amount: float,
    parameters: SimulationParameters,
    resources,
    counter: list[float],
):
    """Transfer payload over one physical H2H edge and record actual bytes."""
    yield env.process(resources.h2h_link[edge].use(amount))
    if parameters.base_die_latency_s > 0.0:
        yield env.timeout(parameters.base_die_latency_s)
    counter.append(amount)


@dataclass
class _CollectiveState:
    done: simpy.Event
    submissions: dict[int, tuple[float, float]]
    started: bool = False


class RingCollectiveCoordinator:
    """Coordinate every rank through globally barriered H2H ring rounds.

    H2H uses explicit ``2*(TP-1)`` rounds. GPU-direct deliberately remains one
    topology-abstract aggregate event; a hybrid logical collective completes
    only after both route abstractions finish.
    """

    def __init__(
        self,
        env: simpy.Environment,
        parameters: SimulationParameters,
        resources,
        counters: H2HTrafficCounters,
    ) -> None:
        self.env = env
        self.parameters = parameters
        self.resources = resources
        self.counters = counters
        self._states: dict[int, _CollectiveState] = {}

    def participate(
        self,
        *,
        collective_index: int,
        requester: int,
        h2h_collective_bytes: float,
        gpu_direct_collective_bytes: float,
    ):
        if collective_index < 0:
            raise ValueError("collective_index cannot be negative")
        if not 0 <= requester < self.parameters.tp:
            raise ValueError("requester must identify a TP rank")
        if h2h_collective_bytes < 0.0 or gpu_direct_collective_bytes < 0.0:
            raise ValueError("collective route bytes cannot be negative")

        state = self._states.setdefault(
            collective_index,
            _CollectiveState(self.env.event(), {}),
        )
        if requester in state.submissions:
            raise ValueError(
                f"rank {requester} submitted collective {collective_index} twice"
            )
        state.submissions[requester] = (
            h2h_collective_bytes,
            gpu_direct_collective_bytes,
        )
        if len(state.submissions) == self.parameters.tp:
            state.started = True
            self.env.process(self._execute(state))
        yield state.done

    def _h2h_route(self, state: _CollectiveState):
        submissions = state.submissions
        source_reads = []
        for requester, (amount, _) in submissions.items():
            if amount <= 0.0:
                continue
            if (
                self.parameters.comm_mode == "cut_through"
                or self.parameters.include_materialized_endpoints
            ):
                source_reads.append(
                    self.env.process(
                        _use_local_hbm(
                            self.env,
                            self.resources,
                            requester,
                            amount,
                            self.parameters,
                        )
                    )
                )
        if source_reads:
            yield self.env.all_of(source_reads)

        round_payloads = {
            requester: _split_collective_round_bytes(amount, self.parameters.tp)
            for requester, (amount, _) in submissions.items()
        }
        round_count = ring_allreduce_round_count(self.parameters.tp)
        for round_index in range(round_count):
            transfers = []
            for requester, payloads in round_payloads.items():
                amount = payloads[round_index]
                if amount <= 0.0:
                    continue
                edge = (requester, (requester + 1) % self.parameters.tp)
                transfers.append(
                    self.env.process(
                        _h2h_edge_transfer(
                            self.env,
                            edge,
                            amount,
                            self.parameters,
                            self.resources,
                            self.counters.collective_edge_amounts,
                        )
                    )
                )
            if transfers:
                # A global barrier separates successive reduce-scatter /
                # all-gather rounds; no rank starts round i+1 early.
                yield self.env.all_of(transfers)

        if (
            self.parameters.comm_mode == "materialized"
            and self.parameters.include_materialized_endpoints
        ):
            # Intermediate rounds stay in the base-die/network abstraction.
            # The final logical result is written and reread from each
            # participant's own local HBM, not the next ring tile's HBM.
            final_writes = [
                self.env.process(
                    _use_local_hbm(
                        self.env,
                        self.resources,
                        requester,
                        amount,
                        self.parameters,
                    )
                )
                for requester, (amount, _) in submissions.items()
                if amount > 0.0
            ]
            if final_writes:
                yield self.env.all_of(final_writes)
            final_rereads = [
                self.env.process(
                    _use_local_hbm(
                        self.env,
                        self.resources,
                        requester,
                        amount,
                        self.parameters,
                    )
                )
                for requester, (amount, _) in submissions.items()
                if amount > 0.0
            ]
            if final_rereads:
                yield self.env.all_of(final_rereads)

    def _direct_route(self, state: _CollectiveState):
        events = []
        for requester, (_, amount) in state.submissions.items():
            if amount <= 0.0:
                continue
            destination = (requester + 1) % self.parameters.tp
            events.append(
                self.env.process(
                    self.resources.gpu_direct_link[(requester, destination)].use(
                        amount
                    )
                )
            )
        if events:
            yield self.env.all_of(events)

    def _execute(self, state: _CollectiveState):
        routes = []
        if any(h2h > 0.0 for h2h, _ in state.submissions.values()):
            routes.append(self.env.process(self._h2h_route(state)))
        if any(direct > 0.0 for _, direct in state.submissions.values()):
            routes.append(self.env.process(self._direct_route(state)))
        if routes:
            yield self.env.all_of(routes)
        state.done.succeed()


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
    coordinator: RingCollectiveCoordinator | None = None,
):
    started = env.now
    if streams.n_messages <= 0 or parameters.tp <= 1:
        intervals[tile] = []
        return

    destination = (tile + 1) % parameters.tp
    h2h_per_message = streams.h2h_bytes / streams.n_messages
    direct_per_message = streams.gpu_direct_bytes / streams.n_messages
    for message_index in range(streams.n_messages):
        yield env.process(
            _communication_message(
                env,
                tile,
                destination,
                h2h_per_message,
                direct_per_message,
                parameters,
                resources,
                coordinator=coordinator,
                collective_index=message_index,
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
    *,
    coordinator: RingCollectiveCoordinator | None = None,
    collective_index: int = 0,
):
    if coordinator is not None:
        yield env.process(
            coordinator.participate(
                collective_index=collective_index,
                requester=source,
                h2h_collective_bytes=h2h_amount,
                gpu_direct_collective_bytes=direct_amount,
            )
        )
        return

    # Retained only as the legacy single-step private helper for focused
    # historical diagnostics. Production run_simulation always supplies the
    # global coordinator above.
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


def _semantic_loose_tile_process(
    env: simpy.Environment,
    tile: int,
    stages: tuple[StageTrafficStreams, ...],
    parameters: SimulationParameters,
    resources,
    communication_intervals: dict[int, list[tuple[float, float]]],
    counters: H2HTrafficCounters | None = None,
    coordinator: RingCollectiveCoordinator | None = None,
):
    """Run semantic stages with the legacy loose collective dependency."""
    destination = (tile + 1) % parameters.tp if parameters.tp > 1 else tile
    communication_events = []
    if counters is None:
        counters = H2HTrafficCounters()
    side_request_index = 0

    def timed_message(stage_index: int, stage: StageTrafficStreams):
        started = env.now
        yield env.process(
            _communication_message(
                env,
                tile,
                destination,
                stage.h2h_bytes,
                stage.gpu_direct_bytes,
                parameters,
                resources,
                coordinator=coordinator,
                collective_index=stage_index,
            )
        )
        communication_intervals.setdefault(tile, []).append((started, env.now))

    for stage_index, stage in enumerate(stages):
        current_side_index = (
            side_request_index if stage.side_hbm_bytes > 0.0 else None
        )
        yield env.process(
            _semantic_stage_memory_process(
                env,
                tile,
                stage,
                parameters,
                resources,
                counters,
                current_side_index,
            )
        )
        if current_side_index is not None:
            side_request_index += 1
        if stage.compute_flops > 0.0:
            yield env.process(
                resources.gpu_compute[tile].use(stage.compute_flops)
            )
        if (
            parameters.tp > 1
            and stage.h2h_bytes + stage.gpu_direct_bytes > 0.0
        ):
            communication_events.append(
                env.process(timed_message(stage_index, stage))
            )

    if communication_events:
        yield env.all_of(communication_events)
    else:
        communication_intervals[tile] = []


def _semantic_strict_tile_process(
    env: simpy.Environment,
    tile: int,
    stages: tuple[StageTrafficStreams, ...],
    parameters: SimulationParameters,
    resources,
    communication_intervals: dict[int, list[tuple[float, float]]],
    counters: H2HTrafficCounters | None = None,
    coordinator: RingCollectiveCoordinator | None = None,
):
    """Run attention/MLP stages with Comm_i gating dependent C_(i+1)."""
    destination = (tile + 1) % parameters.tp if parameters.tp > 1 else tile
    pending_comm = None
    if counters is None:
        counters = H2HTrafficCounters()
    side_request_index = 0

    def timed_message(stage_index: int, stage: StageTrafficStreams):
        started = env.now
        yield env.process(
            _communication_message(
                env,
                tile,
                destination,
                stage.h2h_bytes,
                stage.gpu_direct_bytes,
                parameters,
                resources,
                coordinator=coordinator,
                collective_index=stage_index,
            )
        )
        communication_intervals.setdefault(tile, []).append((started, env.now))

    for stage_index, stage in enumerate(stages):
        current_side_index = (
            side_request_index if stage.side_hbm_bytes > 0.0 else None
        )
        memory_event = env.process(
            _semantic_stage_memory_process(
                env,
                tile,
                stage,
                parameters,
                resources,
                counters,
                current_side_index,
            )
        )
        if current_side_index is not None:
            side_request_index += 1
        wait_events = [memory_event]
        if pending_comm is not None:
            wait_events.append(pending_comm)
        yield env.all_of(wait_events)

        if stage.compute_flops > 0.0:
            yield env.process(
                resources.gpu_compute[tile].use(stage.compute_flops)
            )
        if (
            parameters.tp > 1
            and stage.h2h_bytes + stage.gpu_direct_bytes > 0.0
        ):
            pending_comm = env.process(timed_message(stage_index, stage))
        else:
            pending_comm = None

    if pending_comm is not None:
        yield pending_comm
    else:
        communication_intervals.setdefault(tile, [])


def _semantic_sequential_tile_process(
    env: simpy.Environment,
    tile: int,
    stages: tuple[StageTrafficStreams, ...],
    parameters: SimulationParameters,
    resources,
    communication_intervals: dict[int, list[tuple[float, float]]],
    counters: H2HTrafficCounters | None = None,
    coordinator: RingCollectiveCoordinator | None = None,
):
    """Run semantic stages without cross-stage memory/compute/comm overlap."""
    destination = (tile + 1) % parameters.tp if parameters.tp > 1 else tile
    intervals = communication_intervals.setdefault(tile, [])
    if counters is None:
        counters = H2HTrafficCounters()
    side_request_index = 0
    for stage_index, stage in enumerate(stages):
        current_side_index = (
            side_request_index if stage.side_hbm_bytes > 0.0 else None
        )
        yield env.process(
            _semantic_stage_memory_process(
                env,
                tile,
                stage,
                parameters,
                resources,
                counters,
                current_side_index,
            )
        )
        if current_side_index is not None:
            side_request_index += 1
        if stage.compute_flops > 0.0:
            yield env.process(
                resources.gpu_compute[tile].use(stage.compute_flops)
            )
        if (
            parameters.tp > 1
            and stage.h2h_bytes + stage.gpu_direct_bytes > 0.0
        ):
            started = env.now
            yield env.process(
                _communication_message(
                    env,
                    tile,
                    destination,
                    stage.h2h_bytes,
                    stage.gpu_direct_bytes,
                    parameters,
                    resources,
                    coordinator=coordinator,
                    collective_index=stage_index,
                )
            )
            intervals.append((started, env.now))


def _causal_tile_process(
    env: simpy.Environment,
    tile: int,
    streams: TrafficStreams,
    signature: TrafficSignature,
    parameters: SimulationParameters,
    resources,
    communication_intervals: dict[int, list[tuple[float, float]]],
    counters: H2HTrafficCounters | None = None,
    coordinator: RingCollectiveCoordinator | None = None,
):
    """Ablation / optimistic upper-bound mode that permits execution before the
    prior collective dependency is satisfied; not intended as the default
    causally-correct TP execution model. Event-timeline audit shows this mode
    can start compute_(i+1) before collective_i completes, which is not a
    legitimate independent-operator overlap in the Megatron-style TP sense --
    it removes a real data dependency."""
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
    if counters is None:
        counters = H2HTrafficCounters()

    def timed_message(stage_index: int):
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
                coordinator=coordinator,
                collective_index=stage_index,
            )
        )
        communication_intervals.setdefault(tile, []).append((started, env.now))

    for stage in range(stage_count):
        if parameters.local_side_memory_concurrency == "full_overlap":
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
                    env.process(
                        _side_hbm_read_process(
                            env,
                            tile,
                            side_per_request,
                            stage,
                            parameters,
                            resources,
                            counters,
                        )
                    )
                )
            if memory_events:
                yield env.all_of(memory_events)
        else:
            yield env.process(
                _serialized_local_side_memory_process(
                    env,
                    tile,
                    local_per_stage,
                    side_per_request,
                    1 if stage < streams.n_side_requests else 0,
                    parameters,
                    resources,
                    counters,
                    stage,
                )
            )
        if compute_per_stage > 0:
            yield env.process(resources.gpu_compute[tile].use(compute_per_stage))

        if stage < streams.n_messages and parameters.tp > 1:
            communication_events.append(env.process(timed_message(stage)))

    if communication_events:
        yield env.all_of(communication_events)
    else:
        communication_intervals[tile] = []


def _strict_causal_tile_process(
    env: simpy.Environment,
    tile: int,
    streams: TrafficStreams,
    signature: TrafficSignature,
    parameters: SimulationParameters,
    resources,
    communication_intervals: dict[int, list[tuple[float, float]]],
    counters: H2HTrafficCounters | None = None,
    coordinator: RingCollectiveCoordinator | None = None,
):
    """Run the diagnostic DAG where Comm_i gates C_(i+1), but not M_(i+1).

    Confirmed by event-timeline audit: t(compute_(i+1) start) ==
    t(collective_i end) under this mode, matching Megatron-style TP's
    AllReduce-as-intermediate-synchronization semantics.
    """
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
    pending_comm = None
    if counters is None:
        counters = H2HTrafficCounters()

    def timed_message(stage_index: int):
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
                coordinator=coordinator,
                collective_index=stage_index,
            )
        )
        communication_intervals.setdefault(tile, []).append((started, env.now))

    for stage in range(stage_count):
        if parameters.local_side_memory_concurrency == "full_overlap":
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
                    env.process(
                        _side_hbm_read_process(
                            env,
                            tile,
                            side_per_request,
                            stage,
                            parameters,
                            resources,
                            counters,
                        )
                    )
                )
        else:
            memory_events = [
                env.process(
                    _serialized_local_side_memory_process(
                        env,
                        tile,
                        local_per_stage,
                        side_per_request,
                        1 if stage < streams.n_side_requests else 0,
                        parameters,
                        resources,
                        counters,
                        stage,
                    )
                )
            ]

        wait_events = list(memory_events)
        if pending_comm is not None:
            wait_events.append(pending_comm)
        if wait_events:
            yield env.all_of(wait_events)

        if compute_per_stage > 0:
            yield env.process(resources.gpu_compute[tile].use(compute_per_stage))

        if stage < streams.n_messages and parameters.tp > 1:
            pending_comm = env.process(timed_message(stage))
        else:
            pending_comm = None

    if pending_comm is not None:
        yield pending_comm
    else:
        communication_intervals.setdefault(tile, [])


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
    def snapshot(stats) -> ResourceMetric:
        return ResourceMetric(
            request_count=stats.request_count,
            transferred_amount=stats.transferred_amount,
            total_queue_time_s=stats.total_queue_time_s,
            total_service_time_s=stats.total_service_time_s,
            total_latency_time_s=stats.total_latency_time_s,
            first_service_start_s=stats.first_service_start_s,
            utilization=stats.utilization(elapsed_s),
            by_traffic_class={
                traffic_class: snapshot(class_stats)
                for traffic_class, class_stats in stats.by_traffic_class.items()
            },
        )

    metrics = {}
    for resource in resources.all_resources():
        stats = resource.stats
        metrics[resource.name] = snapshot(stats)
    return metrics


def _materialized_receive_buffer_bytes(
    signature: TrafficSignature,
    parameters: SimulationParameters,
) -> float:
    """Return peak reusable per-GPU scratch space for one H2H round chunk."""
    if (
        parameters.comm_mode != "materialized"
        or not parameters.include_materialized_endpoints
        or parameters.tp <= 1
        or parameters.r >= 1.0
        or signature.n_messages <= 0
        or signature.b_remote_bytes <= 0.0
    ):
        return 0.0
    rounds = ring_allreduce_round_count(parameters.tp)
    if signature.semantic_stages is not None:
        return max(
            (
                (1.0 - parameters.r) * stage.collective_bytes / rounds
                for stage in signature.semantic_stages
            ),
            default=0.0,
        )
    return (
        (1.0 - parameters.r)
        * signature.b_remote_bytes
        / signature.n_messages
        / rounds
    )


def run_simulation(
    signature: TrafficSignature,
    parameters: SimulationParameters,
) -> SimulationResult:
    """Run one representative Stage 2 workload step across a TP ring."""
    if parameters.comm_mode not in {"materialized", "cut_through"}:
        raise ValueError("comm_mode must be materialized or cut_through")
    if parameters.local_side_memory_concurrency not in {
        "full_overlap",
        "serialized",
    }:
        raise ValueError(
            "local_side_memory_concurrency must be full_overlap or serialized"
        )
    _validate_side_hbm_architecture(parameters)
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

    physical_local_capacity = (
        parameters.local_hbm_stack_capacity_bytes
        * parameters.local_hbm_stack_count
    )
    placement_capacity = physical_local_capacity
    local_hbm_fractions = None
    materialized_headroom = _materialized_receive_buffer_bytes(
        signature,
        parameters,
    )
    if parameters.local_hbm_placement_mode == "capacity_overflow":
        local_hbm_fractions = _local_hbm_fractions(parameters)
        # All resident bytes and the receive buffer use the same declarative
        # stack fractions in the downstream capacity check. Uniform mapping
        # therefore exposes the aggregate physical capacity; under weighted
        # mapping the hottest stack binds first at C_stack / max(fraction).
        mapped_local_capacity = (
            physical_local_capacity
            if parameters.local_hbm_mapping_policy == "uniform"
            else (
                parameters.local_hbm_stack_capacity_bytes
                / max(local_hbm_fractions)
            )
        )
        # If the receive buffer alone is too large, leave no resident capacity;
        # the unchanged downstream safety check will report infeasibility.
        placement_capacity = max(
            mapped_local_capacity - materialized_headroom,
            0.0,
        )
        if (
            placement_capacity > 0.0
            and placement_capacity + materialized_headroom
            > mapped_local_capacity
        ):
            # Avoid an exact-fit false overflow caused solely by one-ULP
            # subtract/add cancellation. Cut-through (headroom=0) never enters
            # this branch, preserving its placement bit-for-bit.
            placement_capacity = math.nextafter(placement_capacity, 0.0)

    streams = split_traffic(
        signature,
        parameters.r_remote,
        parameters.r,
        local_hbm_placement_mode=parameters.local_hbm_placement_mode,
        total_local_capacity_bytes=placement_capacity,
        side_hbm_capacity_bytes=parameters.side_hbm_capacity_bytes,
    )
    if local_hbm_fractions is None:
        local_hbm_fractions = _local_hbm_fractions(parameters)
    if streams.side_hbm_bytes > 0.0 and streams.n_side_requests <= 0:
        raise ValueError("nonzero side-HBM traffic requires a positive request count")
    if (
        parameters.tp == 1
        and parameters.side_hbm_path == "pillar_bypass"
        and streams.side_hbm_bytes > 0.0
    ):
        raise ValueError("TP=1 pillar_bypass cannot serve positive side traffic")
    if (
        streams.h2h_bytes + streams.gpu_direct_bytes > 0.0
        and streams.n_messages <= 0
    ):
        raise ValueError("nonzero collective traffic requires a positive request count")
    materialized_buffer = materialized_headroom
    if parameters.local_hbm_placement_mode == "fixed_fraction":
        # Capacity is resident state, not transferred traffic. Legacy/custom
        # signatures without an explicit footprint retain the previous b_local
        # behavior through this fallback. Keep this original fixed-fraction
        # formula bit-exact.
        resident_footprint = (
            signature.b_local_bytes
            if signature.local_footprint_bytes is None
            else signature.local_footprint_bytes
        )
        required_local_capacity = (
            (1.0 - parameters.r_remote) * resident_footprint
            + materialized_buffer
        )
    else:
        # Capacity-overflow placement already moved non-local resident classes
        # to side HBM. Rechecking the pre-placement total would reject valid
        # placements, so only the post-placement local footprint belongs here.
        required_local_capacity = (
            streams.local_resident_footprint_bytes + materialized_buffer
        )
    capacity_remedy = (
        "increase r_remote/r or capacity"
        if parameters.local_hbm_placement_mode == "fixed_fraction"
        else "increase local capacity or reduce the resident footprint/buffer"
    )
    for stack, fraction in enumerate(local_hbm_fractions):
        if (
            required_local_capacity * fraction
            > parameters.local_hbm_stack_capacity_bytes
        ):
            raise LocalHBMCapacityError(
                stack,
                f"local_hbm stack {stack} working set and materialized receive "
                f"buffer exceed per-stack capacity; {capacity_remedy}"
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
        local_hbm_latency_s=parameters.local_hbm_latency_s,
        side_hbm_bw_bps=parameters.side_hbm_bw_bps,
        side_hbm_latency_s=parameters.side_hbm_latency_s,
        h2h_bw_bps=parameters.h2h_bw_bps,
        h2h_latency_s=parameters.h2h_latency_s,
        gpu_direct_bw_bps=parameters.gpu_direct_bw_bps,
        gpu_direct_latency_s=parameters.gpu_direct_latency_s,
        gpu_compute_flops=effective_gpu_flops,
        physical_adjacency=(
            parameters.physical_adjacency
            if parameters.side_hbm_path == "pillar_bypass"
            else None
        ),
        pillar_link_bw_bps=parameters.pillar_link_bw_bps,
        pillar_link_latency_s=parameters.pillar_link_latency_s or 0.0,
    )

    communication_intervals: dict[int, list[tuple[float, float]]] = {}
    completion_times: dict[int, float] = {}
    h2h_counters = H2HTrafficCounters()
    collective_coordinator = RingCollectiveCoordinator(
        env,
        parameters,
        resources,
        h2h_counters,
    )

    def tile_process(tile: int):
        if streams.semantic_stages is not None and parameters.allow_overlap:
            semantic_process = (
                _semantic_strict_tile_process
                if parameters.strict_collective_dependency
                else _semantic_loose_tile_process
            )
            yield env.process(
                semantic_process(
                    env,
                    tile,
                    streams.semantic_stages,
                    parameters,
                    resources,
                    communication_intervals,
                    h2h_counters,
                    collective_coordinator,
                )
            )
        elif streams.semantic_stages is not None:
            yield env.process(
                _semantic_sequential_tile_process(
                    env,
                    tile,
                    streams.semantic_stages,
                    parameters,
                    resources,
                    communication_intervals,
                    h2h_counters,
                    collective_coordinator,
                )
            )
        elif parameters.allow_overlap:
            causal_process = (
                _strict_causal_tile_process
                if parameters.strict_collective_dependency
                else _causal_tile_process
            )
            yield env.process(
                causal_process(
                    env,
                    tile,
                    streams,
                    signature,
                    parameters,
                    resources,
                    communication_intervals,
                    h2h_counters,
                    collective_coordinator,
                )
            )
        else:
            yield env.process(
                _memory_compute_process(
                    env,
                    tile,
                    streams,
                    signature,
                    parameters,
                    resources,
                    h2h_counters,
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
                    collective_coordinator,
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
        elapsed_s=elapsed_s,
        ttft_s=elapsed_s,
        tpot_s=elapsed_s / signature.token_count,
        communication_time_s=communication_time_s,
        overlap_s=overlap_s,
        collective_h2h_edge_bytes=h2h_counters.collective_h2h_edge_bytes,
        side_hbm_read_h2h_edge_bytes=(
            h2h_counters.side_hbm_read_h2h_edge_bytes
        ),
        pillar_side_read_edge_bytes=(
            h2h_counters.pillar_side_read_edge_bytes
        ),
        resource_metrics=_snapshot_metrics(resources, elapsed_s),
    )
