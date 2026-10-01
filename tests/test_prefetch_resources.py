import simpy

from uhbm_simulator.prefetch_config import (
    ArchitectureConfig,
    CapacityMemoryConfig,
    CapacityMemoryDeviceConfig,
    LinkConfig,
    MemoryDeviceConfig,
)


def _memory(capacity=100.0, bw=10.0, latency=0.1):
    return MemoryDeviceConfig(capacity, bw, latency, 1.0)


def _capacity_memory():
    return CapacityMemoryConfig(
        device=CapacityMemoryDeviceConfig(1000.0, 5.0, 0.2, 2.0),
        link=LinkConfig(7.0, 0.3, 3.0, 20.0),
    )


def test_b0_builds_eight_paired_capacity_links_and_2_5d_hbm_lanes():
    """Creating side-tier resources for B0 or fewer than eight lanes must fail."""
    from uhbm_simulator.prefetch_resources import build_prefetch_resources

    env = simpy.Environment()
    architecture = ArchitectureConfig(
        "B0",
        "b0_2_5d",
        8,
        _memory(capacity=48e9, bw=12.0),
    )
    resources = build_prefetch_resources(env, architecture, _capacity_memory())

    assert set(resources.capacity_memory) == set(range(8))
    assert set(resources.capacity_link) == set(range(8))
    assert set(resources.local_memory) == set(range(8))
    assert resources.side_hbm == {}
    assert resources.side_link == {}
    assert resources.local_memory[3].name == "hbm_2_5d[3]"
    assert resources.local_memory[3].storage_capacity == 48e9


def test_b2_builds_eight_uhbm_side_hbm_and_shared_bidirectional_links():
    """Separate read/write side links would hide their FCFS contention."""
    from uhbm_simulator.prefetch_resources import build_prefetch_resources

    env = simpy.Environment()
    architecture = ArchitectureConfig(
        "B2",
        "b2_uhbm",
        8,
        _memory(capacity=12e9, bw=24.0),
        side_hbm=_memory(capacity=48e9, bw=12.0),
        side_link=LinkConfig(9.0, 0.4, 4.0),
    )
    resources = build_prefetch_resources(env, architecture, _capacity_memory())

    assert set(resources.capacity_memory) == set(range(8))
    assert set(resources.capacity_link) == set(range(8))
    assert set(resources.local_memory) == set(range(8))
    assert set(resources.side_hbm) == set(range(8))
    assert set(resources.side_link) == set(range(8))
    assert resources.local_memory[5].name == "uhbm[5]"
    assert resources.side_hbm[5].name == "side_hbm_prefetch[5]"
    assert resources.side_link[5].name == "side_link_bidirectional[5]"


def test_prefetch_resources_do_not_alias_collective_h2h_resources():
    """Accidentally routing prefetch through D-1 links must fail this test."""
    from uhbm_simulator.prefetch_resources import build_prefetch_resources
    from uhbm_simulator.resources import build_resource_layer

    env = simpy.Environment()
    architecture = ArchitectureConfig(
        "B2",
        "b2_uhbm",
        8,
        _memory(),
        side_hbm=_memory(),
        side_link=LinkConfig(9.0, 0.0, 1.0),
    )
    prefetch = build_prefetch_resources(env, architecture, _capacity_memory())
    collective = build_resource_layer(
        env=env,
        tp=2,
        r=0.0,
        local_hbm_stack_count=1,
        local_hbm_stack_bw_bps=10.0,
        local_hbm_stack_capacity_bytes=100.0,
        side_hbm_bw_bps=10.0,
        side_hbm_latency_s=0.0,
        h2h_bw_bps=10.0,
        h2h_latency_s=0.0,
        gpu_direct_bw_bps=10.0,
        gpu_direct_latency_s=0.0,
        gpu_compute_flops=10.0,
    )

    prefetch_ids = {id(resource) for resource in prefetch.all_resources()}
    collective_ids = {id(resource) for resource in collective.h2h_link.values()}
    assert prefetch_ids.isdisjoint(collective_ids)
    assert all(not resource.name.startswith("h2h_link") for resource in prefetch.all_resources())

