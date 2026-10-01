import math

import pytest

from test_layer_prefetch_simulator import _config
from uhbm_simulator.resources import TrafficClass


def _result_and_metrics():
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation
    from uhbm_simulator.prefetch_sweep import calculate_prefetch_metrics

    config = _config()
    result = run_prefetch_simulation(config, detailed_trace=False)
    return config, result, calculate_prefetch_metrics(config, result)


def test_path_metrics_report_actual_bytes_requests_queue_service_and_latency():
    """Inferring path activity from utilization instead of tagged stats fails."""
    _, result, metrics = _result_and_metrics()
    path = metrics.paths[TrafficClass.CAPACITY_MEMORY_READ]

    assert path.transferred_bytes == result.path_bytes[TrafficClass.CAPACITY_MEMORY_READ]
    assert path.request_count > 0
    assert path.queue_time_s >= 0.0
    assert path.service_time_s > 0.0
    assert path.latency_time_s >= 0.0


def test_local_memory_access_classes_remain_separate():
    """Collapsing GPU reads, prefetch writes, and writeback reads must fail."""
    _, _, metrics = _result_and_metrics()

    assert metrics.paths[TrafficClass.UHBM_GPU_READ].transferred_bytes > 0.0
    assert metrics.paths[TrafficClass.UHBM_PREFETCH_WRITE].transferred_bytes > 0.0
    assert metrics.paths[TrafficClass.UHBM_WRITEBACK_READ].transferred_bytes > 0.0


def test_lane_utilization_reports_mean_max_and_critical_lane():
    """A single aggregate utilization would hide the binding lane."""
    _, _, metrics = _result_and_metrics()

    assert 0.0 <= metrics.local_lane_utilization_mean <= 1.0
    assert metrics.local_lane_utilization_mean <= metrics.local_lane_utilization_max
    assert metrics.critical_local_lane.startswith("rank[")
    assert ".uhbm[" in metrics.critical_local_lane


def test_energy_uses_actual_bytes_and_conserves_path_sum():
    """Service-time-derived energy or an omitted path must fail conservation."""
    config, _, metrics = _result_and_metrics()
    path = metrics.paths[TrafficClass.CAPACITY_MEMORY_READ]
    expected = (
        path.transferred_bytes
        * 8.0
        * config.capacity_memory.device.read_energy_pj_per_bit
        * 1e-12
    )

    assert path.energy_j == pytest.approx(expected)
    assert metrics.total_dynamic_energy_j == pytest.approx(
        math.fsum(item.energy_j for item in metrics.paths.values())
    )


def test_bidirectional_side_link_accounting_has_no_crossing_double_count():
    """Charging side-link bytes again as side-HBM datapath energy must fail."""
    _, _, metrics = _result_and_metrics()
    prefetch = metrics.paths[TrafficClass.SIDE_TO_UHBM]
    writeback = metrics.paths[TrafficClass.UHBM_TO_SIDE]
    side_reads = metrics.paths[TrafficClass.SIDE_HBM_READ]
    side_writes = metrics.paths[TrafficClass.SIDE_HBM_WRITE]

    assert prefetch.transferred_bytes > 0.0
    assert writeback.transferred_bytes > 0.0
    assert metrics.energy_by_domain_j["side_link"] == pytest.approx(
        prefetch.energy_j + writeback.energy_j
    )
    assert metrics.energy_by_domain_j["side_hbm"] == pytest.approx(
        side_reads.energy_j + side_writes.energy_j
    )


def test_collective_transport_metrics_and_energy_use_collective_resources():
    """Omitting D-1 collective paths from the dedicated report must fail."""
    config, _, metrics = _result_and_metrics()
    h2h = metrics.paths[TrafficClass.COLLECTIVE_H2H_TRANSPORT]

    assert h2h.transferred_bytes > 0.0
    assert h2h.request_count > 0
    assert h2h.energy_j == pytest.approx(
        h2h.transferred_bytes
        * 8.0
        * config.execution.h2h_energy_pj_per_bit
        * 1e-12
    )
    assert metrics.energy_by_domain_j["collective_h2h"] == pytest.approx(
        h2h.energy_j
    )


def test_collective_local_endpoint_energy_uses_local_memory_anchor():
    """Collective source/endpoint accesses must not silently get zero energy."""
    config, _, metrics = _result_and_metrics()
    source_read = metrics.paths[TrafficClass.COLLECTIVE_SOURCE_READ]

    assert source_read.transferred_bytes > 0.0
    assert source_read.energy_j == pytest.approx(
        source_read.transferred_bytes
        * 8.0
        * config.architectures[0].local_memory.energy_pj_per_bit
        * 1e-12
    )


def test_prefetch_production_paths_have_no_unclassified_transfer_requests():
    """Every memory or transport request in the new engine must be tagged."""
    _, result, _ = _result_and_metrics()
    offenders = {}
    for resource in result.resources:
        if "gpu_compute" in resource.name:
            continue
        unclassified = resource.stats.by_traffic_class.get(
            TrafficClass.UNCLASSIFIED
        )
        if unclassified is not None and unclassified.request_count:
            offenders[resource.name] = unclassified.request_count

    assert offenders == {}
