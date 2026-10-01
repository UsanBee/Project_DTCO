"""Eight-lane resources and chunked transfer primitives for prefetch studies."""

from __future__ import annotations

from dataclasses import dataclass
import math

import simpy

from .prefetch_config import ArchitectureConfig, CapacityMemoryConfig
from .resources import RateResource


@dataclass
class PrefetchResourceLayer:
    """Physical resources belonging only to the paired prefetch hierarchy."""

    capacity_memory: dict[int, RateResource]
    capacity_link: dict[int, RateResource]
    local_memory: dict[int, RateResource]
    side_hbm: dict[int, RateResource]
    side_link: dict[int, RateResource]

    def all_resources(self) -> list[RateResource]:
        resources: list[RateResource] = []
        for group in (
            self.capacity_memory,
            self.capacity_link,
            self.local_memory,
            self.side_hbm,
            self.side_link,
        ):
            resources.extend(group.values())
        return resources


@dataclass(frozen=True)
class ChunkTransferResult:
    """Completed chunk sequence and its wall-clock completion time."""

    chunk_amounts: tuple[float, ...]
    completed_at_s: float


def _chunk_amounts(total_bytes: float, chunk_bytes: float) -> tuple[float, ...]:
    if not math.isfinite(total_bytes) or total_bytes < 0.0:
        raise ValueError("total_bytes must be nonnegative and finite")
    if not math.isfinite(chunk_bytes) or chunk_bytes <= 0.0:
        raise ValueError("chunk_bytes must be positive and finite")
    if total_bytes == 0.0:
        return ()
    count = int(math.ceil(total_bytes / chunk_bytes))
    if count == 1:
        return (total_bytes,)
    amounts = [chunk_bytes] * (count - 1)
    amounts.append(total_bytes - math.fsum(amounts))
    return tuple(amounts)


def transfer_chunks(
    env: simpy.Environment,
    *,
    total_bytes: float,
    chunk_bytes: float,
    source: RateResource,
    link: RateResource,
    destination: RateResource,
    source_traffic_class: str,
    link_traffic_class: str,
    destination_traffic_class: str,
    max_inflight_chunks: int | None,
    inflight_slots: simpy.Resource | None = None,
):
    """Move chunks through source, link, and destination pipeline stages.

    A finite depth holds its descriptor until the destination stage finishes;
    depth one therefore serializes whole chunks.  Callers may pass one shared
    ``inflight_slots`` resource for all lane-local flows.  Otherwise a private
    pool is created for this transfer, while ``None`` enqueues every chunk.
    """
    amounts = _chunk_amounts(total_bytes, chunk_bytes)
    if max_inflight_chunks is not None and (
        isinstance(max_inflight_chunks, bool)
        or not isinstance(max_inflight_chunks, int)
        or max_inflight_chunks <= 0
    ):
        raise ValueError("max_inflight_chunks must be positive or None")
    if inflight_slots is not None and max_inflight_chunks is not None:
        raise ValueError(
            "pass either shared inflight_slots or max_inflight_chunks, not both"
        )
    slots = inflight_slots
    if slots is None and max_inflight_chunks is not None:
        slots = simpy.Resource(env, capacity=max_inflight_chunks)

    def run_path(amount: float):
        yield env.process(source.use(amount, source_traffic_class))
        yield env.process(link.use(amount, link_traffic_class))
        yield env.process(destination.use(amount, destination_traffic_class))

    def run_chunk(amount: float):
        if slots is None:
            yield from run_path(amount)
            return
        with slots.request() as request:
            yield request
            yield from run_path(amount)

    processes = [env.process(run_chunk(amount)) for amount in amounts]
    if processes:
        yield env.all_of(processes)
    return ChunkTransferResult(amounts, env.now)


def build_prefetch_resources(
    env: simpy.Environment,
    architecture: ArchitectureConfig,
    capacity_memory: CapacityMemoryConfig,
) -> PrefetchResourceLayer:
    """Create one independent paired-memory fabric for each of eight lanes."""
    if architecture.lane_count != 8:
        raise ValueError("prefetch resource layer requires exactly eight lanes")
    if architecture.kind not in {"b0_2_5d", "b2_uhbm"}:
        raise ValueError(f"unknown prefetch architecture kind: {architecture.kind}")

    lanes = range(architecture.lane_count)
    capacity_resources = {
        lane: RateResource(
            env,
            f"capacity_memory[{lane}]",
            capacity_memory.device.read_bw_bps,
            latency_s=capacity_memory.device.read_latency_s,
            storage_capacity=capacity_memory.device.capacity_bytes,
        )
        for lane in lanes
    }
    capacity_links = {
        lane: RateResource(
            env,
            f"capacity_link[{lane}]",
            capacity_memory.link.bw_bps,
            latency_s=capacity_memory.link.latency_s,
        )
        for lane in lanes
    }
    local_prefix = "hbm_2_5d" if architecture.kind == "b0_2_5d" else "uhbm"
    local_resources = {
        lane: RateResource(
            env,
            f"{local_prefix}[{lane}]",
            architecture.local_memory.bw_bps,
            latency_s=architecture.local_memory.latency_s,
            storage_capacity=architecture.local_memory.capacity_bytes,
        )
        for lane in lanes
    }

    side_hbm: dict[int, RateResource] = {}
    side_link: dict[int, RateResource] = {}
    if architecture.kind == "b2_uhbm":
        if architecture.side_hbm is None or architecture.side_link is None:
            raise ValueError("b2_uhbm requires paired side_hbm and side_link")
        side_hbm = {
            lane: RateResource(
                env,
                f"side_hbm_prefetch[{lane}]",
                architecture.side_hbm.bw_bps,
                latency_s=architecture.side_hbm.latency_s,
                storage_capacity=architecture.side_hbm.capacity_bytes,
            )
            for lane in lanes
        }
        # A single lane-local resource is intentionally shared by prefetch and
        # writeback directions.  They compete FCFS instead of receiving an
        # architecture-time fixed bandwidth partition.
        side_link = {
            lane: RateResource(
                env,
                f"side_link_bidirectional[{lane}]",
                architecture.side_link.bw_bps,
                latency_s=architecture.side_link.latency_s,
            )
            for lane in lanes
        }

    return PrefetchResourceLayer(
        capacity_memory=capacity_resources,
        capacity_link=capacity_links,
        local_memory=local_resources,
        side_hbm=side_hbm,
        side_link=side_link,
    )
