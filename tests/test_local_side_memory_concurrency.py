from dataclasses import replace

import pytest

from uhbm_simulator.simulator import SimulationParameters, run_simulation
from uhbm_simulator.workload import TrafficSignature


def parameters(**overrides):
    base = SimulationParameters(
        tp=1,
        r=0.0,
        r_remote=0.5,
        comm_mode="materialized",
        thermal_scenario="no_throttling",
        local_hbm_stack_count=1,
        local_hbm_stack_bw_bps=100.0,
        local_hbm_stack_capacity_bytes=10_000.0,
        local_hbm_mapping_policy="uniform",
        local_hbm_mapping_weights=None,
        side_hbm_bw_bps=100.0,
        side_hbm_latency_s=0.0,
        h2h_bw_bps=100.0,
        h2h_latency_s=0.0,
        gpu_direct_bw_bps=100.0,
        gpu_direct_latency_s=0.0,
        gpu_peak_flops=100.0,
        base_die_latency_s=0.0,
        include_materialized_endpoints=False,
        allow_overlap=True,
        strict_collective_dependency=True,
        # These tests isolate the C-2 local/side concurrency semantics; D-1's
        # shared H2H transport is covered independently.
        side_hbm_path="independent",
    )
    return replace(base, **overrides)


def single_stage_signature():
    return TrafficSignature(
        phase="decode",
        b_local_bytes=200.0,
        b_remote_bytes=0.0,
        n_messages=0,
        compute_flops=0.0,
        n_side_requests=1,
        local_footprint_bytes=200.0,
    )


def multi_stage_signature():
    return TrafficSignature(
        phase="decode",
        b_local_bytes=400.0,
        b_remote_bytes=200.0,
        n_messages=2,
        compute_flops=200.0,
        n_side_requests=2,
        local_footprint_bytes=400.0,
    )


def test_full_overlap_is_default_and_matches_explicit_mode_exactly():
    default_parameters = parameters()
    explicit_parameters = replace(
        default_parameters,
        local_side_memory_concurrency="full_overlap",
    )

    assert default_parameters.local_side_memory_concurrency == "full_overlap"
    assert run_simulation(
        single_stage_signature(), default_parameters
    ) == run_simulation(single_stage_signature(), explicit_parameters)


def test_serialized_single_stage_adds_local_and_side_service_times():
    full_overlap = run_simulation(single_stage_signature(), parameters())
    serialized = run_simulation(
        single_stage_signature(),
        parameters(local_side_memory_concurrency="serialized"),
    )

    assert full_overlap.elapsed_s == 1.0
    assert serialized.elapsed_s == 2.0
    assert full_overlap.resource_metrics["local_hbm[0][0]"].first_service_start_s == 0.0
    assert full_overlap.resource_metrics["side_hbm[0]"].first_service_start_s == 0.0
    assert serialized.resource_metrics["local_hbm[0][0]"].first_service_start_s == 0.0
    assert serialized.resource_metrics["side_hbm[0]"].first_service_start_s == 1.0


def test_serialized_accepts_local_only_traffic():
    signature = replace(
        single_stage_signature(),
        b_local_bytes=100.0,
        n_side_requests=0,
        local_footprint_bytes=100.0,
    )

    result = run_simulation(
        signature,
        parameters(
            r_remote=0.0,
            local_side_memory_concurrency="serialized",
        ),
    )

    assert result.elapsed_s == 1.0


@pytest.mark.parametrize("strict_collective_dependency", [True, False])
def test_serialized_multi_stage_preserves_side_request_count(
    strict_collective_dependency,
):
    signature = replace(
        multi_stage_signature(),
        b_remote_bytes=0.0,
        compute_flops=0.0,
    )

    result = run_simulation(
        signature,
        parameters(
            local_side_memory_concurrency="serialized",
            strict_collective_dependency=strict_collective_dependency,
        ),
    )

    assert result.elapsed_s == 4.0
    assert result.resource_metrics["local_hbm[0][0]"].request_count == 2
    assert result.resource_metrics["side_hbm[0]"].request_count == 2


def test_strict_serialized_memory_still_overlaps_previous_communication():
    result = run_simulation(
        multi_stage_signature(),
        parameters(
            tp=2,
            local_side_memory_concurrency="serialized",
        ),
    )

    # Per tile: M0=2s, C0=1s, then M1=2s overlaps Comm0=1s; C1=1s and
    # final Comm1=1s. Serializing Comm0 before M1 would incorrectly yield 8s.
    assert result.elapsed_s == 7.0
    assert result.communication_time_s == 2.0


def test_overlap_counterfactual_uses_the_same_serialized_memory_mode():
    result = run_simulation(
        multi_stage_signature(),
        parameters(
            tp=2,
            local_side_memory_concurrency="serialized",
        ),
    )

    # The allow_overlap=False rerun takes 8s with serialized local/side memory,
    # so the strict causal run's saved-time metric is 8 - 7 = 1s.
    assert result.elapsed_s == 7.0
    assert result.overlap_s == 1.0


def test_unknown_local_side_memory_concurrency_is_rejected():
    with pytest.raises(ValueError, match="local_side_memory_concurrency"):
        run_simulation(
            single_stage_signature(),
            parameters(local_side_memory_concurrency="partial"),
        )
