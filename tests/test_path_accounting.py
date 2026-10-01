import math

import pytest
import simpy

from uhbm_simulator import resources as resource_module
from uhbm_simulator import simulator as simulator_module


def test_untagged_use_preserves_legacy_totals():
    """Removing the compatibility default must break this legacy contract."""
    env = simpy.Environment()
    resource = resource_module.RateResource(
        env, "link", rate_per_s=10.0, latency_s=1.0
    )

    env.process(resource.use(20.0))
    env.run()

    assert resource.stats.request_count == 1
    assert resource.stats.transferred_amount == 20.0
    assert resource.stats.total_queue_time_s == 0.0
    assert resource.stats.total_service_time_s == 2.0
    assert resource.stats.total_latency_time_s == 1.0
    untagged = resource.stats.by_traffic_class[
        resource_module.TrafficClass.UNCLASSIFIED
    ]
    assert untagged.request_count == 1
    assert untagged.transferred_amount == 20.0


def test_tagged_requests_attribute_bytes_and_wait_time_to_the_request_class():
    """Charging queue time to the wrong traffic class must fail this test."""
    env = simpy.Environment()
    resource = resource_module.RateResource(
        env, "link", rate_per_s=10.0, latency_s=1.0
    )

    env.process(
        resource.use(
            20.0,
            traffic_class=resource_module.TrafficClass.CAPACITY_MEMORY_READ,
        )
    )
    env.process(
        resource.use(10.0, traffic_class=resource_module.TrafficClass.CAPACITY_TO_UHBM)
    )
    env.run()

    memory = resource.stats.by_traffic_class[
        resource_module.TrafficClass.CAPACITY_MEMORY_READ
    ]
    transport = resource.stats.by_traffic_class[
        resource_module.TrafficClass.CAPACITY_TO_UHBM
    ]
    assert memory.transferred_amount == 20.0
    assert memory.total_queue_time_s == 0.0
    assert memory.total_service_time_s == 2.0
    assert memory.total_latency_time_s == 1.0
    assert transport.transferred_amount == 10.0
    assert transport.total_queue_time_s == 2.0
    assert transport.total_service_time_s == 1.0
    assert transport.total_latency_time_s == 1.0


def test_tagged_subtotals_conserve_resource_totals():
    """Dropping or double-counting any tagged request must fail conservation."""
    env = simpy.Environment()
    resource = resource_module.RateResource(
        env, "link", rate_per_s=8.0, latency_s=0.25
    )
    requests = (
        (8.0, resource_module.TrafficClass.SIDE_HBM_READ),
        (12.0, resource_module.TrafficClass.SIDE_TO_UHBM),
        (4.0, resource_module.TrafficClass.UHBM_TO_SIDE),
    )
    for amount, traffic_class in requests:
        env.process(resource.use(amount, traffic_class=traffic_class))
    env.run()

    subtotals = tuple(resource.stats.by_traffic_class.values())
    assert sum(item.request_count for item in subtotals) == 3
    assert math.fsum(item.transferred_amount for item in subtotals) == 24.0
    assert math.fsum(item.total_queue_time_s for item in subtotals) == 3.5
    assert math.fsum(item.total_service_time_s for item in subtotals) == 3.0
    assert math.fsum(item.total_latency_time_s for item in subtotals) == 0.75
    assert math.fsum(item.transferred_amount for item in subtotals) == (
        resource.stats.transferred_amount
    )
    assert math.fsum(item.total_service_time_s for item in subtotals) == (
        resource.stats.total_service_time_s
    )


def test_unknown_explicit_traffic_class_is_rejected():
    """Silently accepting a typo would create an unreportable path."""
    env = simpy.Environment()
    resource = resource_module.RateResource(env, "link", rate_per_s=1.0)

    with pytest.raises(ValueError, match="unknown traffic_class"):
        env.process(resource.use(1.0, traffic_class="capacity_memroy_read"))


def test_resource_metric_exposes_tagged_snapshots_without_changing_totals():
    """Dropping class detail during the immutable snapshot must fail."""
    env = simpy.Environment()
    resource = resource_module.RateResource(env, "link", rate_per_s=4.0)
    env.process(
        resource.use(
            8.0,
            traffic_class=resource_module.TrafficClass.CAPACITY_MEMORY_READ,
        )
    )
    env.run()
    layer = resource_module.ResourceLayer(
        local_hbm={(0, 0): resource},
        side_hbm={},
        h2h_link={},
        pillar_link={},
        gpu_direct_link={},
        gpu_compute={},
    )

    metric = simulator_module._snapshot_metrics(layer, env.now)["link"]

    assert metric.transferred_amount == 8.0
    detail = metric.by_traffic_class[
        resource_module.TrafficClass.CAPACITY_MEMORY_READ
    ]
    assert detail.transferred_amount == 8.0
    assert detail.total_service_time_s == 2.0
