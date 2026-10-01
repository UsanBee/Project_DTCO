import math

import pytest
import simpy

from uhbm_simulator.resources import RateResource, TrafficClass


def _path(env, *, rate=1.0, latency=0.0):
    return (
        RateResource(env, "source", rate, latency_s=latency),
        RateResource(env, "link", rate, latency_s=latency),
        RateResource(env, "destination", rate, latency_s=latency),
    )


def _transfer(env, resources, **overrides):
    from uhbm_simulator.prefetch_resources import transfer_chunks

    kwargs = dict(
        total_bytes=2.0,
        chunk_bytes=1.0,
        source=resources[0],
        link=resources[1],
        destination=resources[2],
        source_traffic_class=TrafficClass.CAPACITY_MEMORY_READ,
        link_traffic_class=TrafficClass.CAPACITY_TO_UHBM,
        destination_traffic_class=TrafficClass.UHBM_PREFETCH_WRITE,
        max_inflight_chunks=1,
    )
    kwargs.update(overrides)
    return env.process(transfer_chunks(env, **kwargs))


def test_chunk_residual_conserves_bytes_on_every_path_segment():
    """Dropping a non-divisible final chunk must fail byte conservation."""
    env = simpy.Environment()
    resources = _path(env, rate=10.0)
    process = _transfer(
        env,
        resources,
        total_bytes=10.1,
        chunk_bytes=3.0,
        max_inflight_chunks=None,
    )
    env.run()

    assert process.value.chunk_amounts[:3] == (3.0, 3.0, 3.0)
    assert process.value.chunk_amounts[3] == pytest.approx(1.1)
    assert math.fsum(process.value.chunk_amounts) == 10.1
    assert [resource.stats.request_count for resource in resources] == [4, 4, 4]
    assert [resource.stats.transferred_amount for resource in resources] == pytest.approx(
        [10.1, 10.1, 10.1]
    )


def test_depth_two_pipelines_source_link_and_destination():
    """Serializing whole chunks despite depth > 1 must fail this timing."""
    env = simpy.Environment()
    process = _transfer(
        env,
        _path(env),
        total_bytes=2.0,
        chunk_bytes=1.0,
        max_inflight_chunks=2,
    )
    env.run()

    assert process.value.completed_at_s == pytest.approx(4.0)


def test_depth_one_disables_inter_chunk_pipeline():
    """Releasing a descriptor before destination completion must fail."""
    env = simpy.Environment()
    process = _transfer(
        env,
        _path(env),
        total_bytes=2.0,
        chunk_bytes=1.0,
        max_inflight_chunks=1,
    )
    env.run()

    assert process.value.completed_at_s == pytest.approx(6.0)


def test_unbounded_depth_enqueues_every_chunk_and_reaches_pipeline_limit():
    """Treating null depth as one or as an arbitrary finite cap must fail."""
    env = simpy.Environment()
    process = _transfer(
        env,
        _path(env),
        total_bytes=4.0,
        chunk_bytes=1.0,
        max_inflight_chunks=None,
    )
    env.run()

    assert process.value.completed_at_s == pytest.approx(6.0)


def test_fixed_latency_is_applied_once_per_chunk_on_each_segment():
    """Applying latency once per aggregate transfer instead of per chunk fails."""
    env = simpy.Environment()
    resources = _path(env, rate=1.0, latency=0.5)
    process = _transfer(
        env,
        resources,
        total_bytes=2.0,
        chunk_bytes=1.0,
        max_inflight_chunks=1,
    )
    env.run()

    assert process.value.completed_at_s == pytest.approx(9.0)
    assert [resource.stats.total_latency_time_s for resource in resources] == [
        1.0,
        1.0,
        1.0,
    ]


def test_prefetch_and_writeback_compete_fcfs_on_one_shared_side_link():
    """Independent directional links would incorrectly remove queueing."""
    from uhbm_simulator.prefetch_resources import transfer_chunks

    env = simpy.Environment()
    shared_link = RateResource(env, "side_link", 1.0)
    prefetch_source = RateResource(env, "side_read", 1e12)
    prefetch_dest = RateResource(env, "uhbm_write", 1e12)
    writeback_source = RateResource(env, "uhbm_read", 1e12)
    writeback_dest = RateResource(env, "side_write", 1e12)

    prefetch = env.process(
        transfer_chunks(
            env,
            total_bytes=1.0,
            chunk_bytes=1.0,
            source=prefetch_source,
            link=shared_link,
            destination=prefetch_dest,
            source_traffic_class=TrafficClass.SIDE_HBM_READ,
            link_traffic_class=TrafficClass.SIDE_TO_UHBM,
            destination_traffic_class=TrafficClass.UHBM_PREFETCH_WRITE,
            max_inflight_chunks=None,
        )
    )
    writeback = env.process(
        transfer_chunks(
            env,
            total_bytes=1.0,
            chunk_bytes=1.0,
            source=writeback_source,
            link=shared_link,
            destination=writeback_dest,
            source_traffic_class=TrafficClass.UHBM_WRITEBACK_READ,
            link_traffic_class=TrafficClass.UHBM_TO_SIDE,
            destination_traffic_class=TrafficClass.SIDE_HBM_WRITE,
            max_inflight_chunks=None,
        )
    )
    env.run()

    assert prefetch.value.completed_at_s < writeback.value.completed_at_s
    assert shared_link.stats.request_count == 2
    assert shared_link.stats.by_traffic_class[
        TrafficClass.UHBM_TO_SIDE
    ].total_queue_time_s == pytest.approx(1.0)

