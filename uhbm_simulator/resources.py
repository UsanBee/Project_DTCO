"""SimPy resource primitives for the Stage 2 system/resource layer."""

from __future__ import annotations

from dataclasses import dataclass

import simpy


@dataclass
class ResourceStats:
    request_count: int = 0
    transferred_amount: float = 0.0
    total_queue_time_s: float = 0.0
    total_service_time_s: float = 0.0
    total_latency_time_s: float = 0.0
    first_service_start_s: float | None = None

    def utilization(self, elapsed_s: float, capacity: int = 1) -> float:
        if elapsed_s <= 0:
            return 0.0
        return min(1.0, self.total_service_time_s / (elapsed_s * capacity))


class RateResource:
    """A serialized SimPy resource serviced at a fixed rate plus latency."""

    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        rate_per_s: float,
        latency_s: float = 0.0,
        capacity: int = 1,
        storage_capacity: float | None = None,
    ) -> None:
        if rate_per_s <= 0:
            raise ValueError("rate_per_s must be positive")
        if latency_s < 0:
            raise ValueError("latency_s cannot be negative")
        self.env = env
        self.name = name
        self.rate_per_s = rate_per_s
        self.latency_s = latency_s
        self.storage_capacity = storage_capacity
        self._resource = simpy.Resource(env, capacity=capacity)
        self.stats = ResourceStats()

    def use(self, amount: float):
        if amount < 0:
            raise ValueError("resource amount cannot be negative")
        queued_at = self.env.now
        with self._resource.request() as request:
            yield request
            queue_time = self.env.now - queued_at
            service_time = amount / self.rate_per_s
            self.stats.request_count += 1
            self.stats.transferred_amount += amount
            self.stats.total_queue_time_s += queue_time
            self.stats.total_service_time_s += service_time
            self.stats.total_latency_time_s += self.latency_s
            if self.stats.first_service_start_s is None:
                self.stats.first_service_start_s = self.env.now
            yield self.env.timeout(service_time)
        if self.latency_s:
            yield self.env.timeout(self.latency_s)


@dataclass
class ResourceLayer:
    local_hbm: dict[tuple[int, int], RateResource]
    side_hbm: dict[int, RateResource]
    h2h_link: dict[tuple[int, int], RateResource]
    gpu_direct_link: dict[tuple[int, int], RateResource]
    gpu_compute: dict[int, RateResource]

    def all_resources(self) -> list[RateResource]:
        resources: list[RateResource] = []
        for group in (
            self.local_hbm,
            self.side_hbm,
            self.h2h_link,
            self.gpu_direct_link,
            self.gpu_compute,
        ):
            resources.extend(group.values())
        return resources


def _ring_edges(tp: int) -> list[tuple[int, int]]:
    if tp <= 1:
        return []
    return [(tile, (tile + 1) % tp) for tile in range(tp)]


def build_resource_layer(
    *,
    env: simpy.Environment,
    tp: int,
    r: float,
    local_hbm_stack_count: int,
    local_hbm_stack_bw_bps: float,
    local_hbm_stack_capacity_bytes: float,
    side_hbm_bw_bps: float,
    side_hbm_latency_s: float,
    h2h_bw_bps: float,
    h2h_latency_s: float,
    gpu_direct_bw_bps: float,
    gpu_direct_latency_s: float,
    gpu_compute_flops: float,
) -> ResourceLayer:
    """Create the five resource groups from spec chapter 8."""
    if tp < 1:
        raise ValueError("tp must be positive")
    if local_hbm_stack_count < 1:
        raise ValueError("local_hbm_stack_count must be positive")
    if not 0.0 <= r <= 1.0:
        raise ValueError("r must be within [0, 1]")

    local_hbm = {
        (tile, stack): RateResource(
            env,
            f"local_hbm[{tile}][{stack}]",
            local_hbm_stack_bw_bps,
            storage_capacity=local_hbm_stack_capacity_bytes,
        )
        for tile in range(tp)
        for stack in range(local_hbm_stack_count)
    }
    side_hbm = {
        tile: RateResource(
            env,
            f"side_hbm[{tile}]",
            side_hbm_bw_bps,
            latency_s=side_hbm_latency_s,
        )
        for tile in range(tp)
    }
    gpu_compute = {
        tile: RateResource(env, f"gpu_compute[{tile}]", gpu_compute_flops)
        for tile in range(tp)
    }

    edges = _ring_edges(tp)
    h2h_link = {
        edge: RateResource(
            env,
            f"h2h_link[{edge[0]},{edge[1]}]",
            h2h_bw_bps,
            latency_s=h2h_latency_s,
        )
        for edge in edges
    }
    gpu_direct_link = (
        {
            edge: RateResource(
                env,
                f"gpu_direct_link[{edge[0]},{edge[1]}]",
                gpu_direct_bw_bps,
                latency_s=gpu_direct_latency_s,
            )
            for edge in edges
        }
        if r > 0.0
        else {}
    )
    return ResourceLayer(
        local_hbm=local_hbm,
        side_hbm=side_hbm,
        h2h_link=h2h_link,
        gpu_direct_link=gpu_direct_link,
        gpu_compute=gpu_compute,
    )
