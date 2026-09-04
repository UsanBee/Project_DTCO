import simpy

from uhbm_simulator.resources import RateResource, build_resource_layer


def test_rate_resource_accounts_for_latency_service_and_queueing():
    env = simpy.Environment()
    resource = RateResource(env, "link", rate_per_s=10.0, latency_s=1.0)

    completions = []

    def transfer():
        yield env.process(resource.use(20.0))
        completions.append(env.now)

    env.process(transfer())
    env.process(transfer())
    env.run()

    assert completions == [3.0, 5.0]
    assert resource.stats.request_count == 2
    assert resource.stats.total_queue_time_s == 2.0
    assert resource.stats.total_service_time_s == 4.0
    assert resource.stats.total_latency_time_s == 2.0
    assert resource.stats.utilization(env.now) == 0.8


def test_resource_layer_has_per_tile_resources_and_no_direct_link_at_r_zero():
    env = simpy.Environment()
    resources = build_resource_layer(
        env=env,
        tp=4,
        r=0.0,
        local_hbm_stack_count=4,
        local_hbm_stack_bw_bps=1.2e12,
        local_hbm_stack_capacity_bytes=16e9,
        side_hbm_bw_bps=4.8e12,
        side_hbm_latency_s=50e-9,
        h2h_bw_bps=900e9,
        h2h_latency_s=20e-9,
        gpu_direct_bw_bps=900e9,
        gpu_direct_latency_s=20e-9,
        gpu_compute_flops=1e15,
    )

    assert set(resources.local_hbm) == {
        (tile, stack) for tile in range(4) for stack in range(4)
    }
    assert resources.local_hbm[(2, 3)].name == "local_hbm[2][3]"
    assert set(resources.side_hbm) == {0, 1, 2, 3}
    assert set(resources.gpu_compute) == {0, 1, 2, 3}
    assert len(resources.h2h_link) == 4
    assert resources.gpu_direct_link == {}


def test_resource_layer_creates_ring_direct_links_only_when_r_is_positive():
    env = simpy.Environment()
    resources = build_resource_layer(
        env=env,
        tp=2,
        r=0.25,
        local_hbm_stack_count=4,
        local_hbm_stack_bw_bps=1.2e12,
        local_hbm_stack_capacity_bytes=16e9,
        side_hbm_bw_bps=4.8e12,
        side_hbm_latency_s=50e-9,
        h2h_bw_bps=900e9,
        h2h_latency_s=20e-9,
        gpu_direct_bw_bps=900e9,
        gpu_direct_latency_s=20e-9,
        gpu_compute_flops=1e15,
    )

    assert set(resources.gpu_direct_link) == {(0, 1), (1, 0)}
