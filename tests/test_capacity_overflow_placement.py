from dataclasses import replace
import math

import pytest

import uhbm_simulator.workload as workload_module
from uhbm_simulator.simulator import SimulationParameters, run_simulation
from uhbm_simulator.workload import (
    TrafficSignature,
    build_decode_signature,
    build_prefill_signature,
    split_traffic,
)


def parameters(**overrides):
    base = SimulationParameters(
        tp=1,
        r=0.0,
        r_remote=0.25,
        comm_mode="materialized",
        thermal_scenario="no_throttling",
        local_hbm_stack_count=1,
        local_hbm_stack_bw_bps=100.0,
        local_hbm_stack_capacity_bytes=1_000.0,
        local_hbm_mapping_policy="uniform",
        local_hbm_mapping_weights=None,
        side_hbm_bw_bps=100.0,
        side_hbm_latency_s=0.0,
        h2h_bw_bps=100.0,
        h2h_latency_s=0.0,
        gpu_direct_bw_bps=100.0,
        gpu_direct_latency_s=0.0,
        gpu_peak_flops=1e12,
        base_die_latency_s=0.0,
        allow_overlap=False,
    )
    return replace(base, **overrides)


def legacy_signature():
    return TrafficSignature(
        phase="decode",
        b_local_bytes=100.0,
        b_remote_bytes=0.0,
        n_messages=0,
        compute_flops=0.0,
        n_side_requests=1,
        local_footprint_bytes=40.0,
    )


def placement_signature(
    *,
    weight_footprint=40.0,
    kv_footprint=40.0,
    activation_footprint=20.0,
    weight_traffic=100.0,
    kv_traffic=10.0,
    activation_traffic=1_000.0,
):
    return TrafficSignature(
        phase="decode",
        b_local_bytes=weight_traffic + kv_traffic + activation_traffic,
        b_remote_bytes=0.0,
        n_messages=0,
        compute_flops=0.0,
        n_side_requests=1,
        local_footprint_bytes=(
            weight_footprint + kv_footprint + activation_footprint
        ),
        weight_footprint_bytes=weight_footprint,
        kv_footprint_bytes=kv_footprint,
        activation_footprint_bytes=activation_footprint,
        weight_traffic_bytes=weight_traffic,
        kv_traffic_bytes=kv_traffic,
        activation_traffic_bytes_value=activation_traffic,
    )


def materialized_spill_signature(*, total_footprint=80.0):
    return replace(
        placement_signature(
            weight_footprint=60.0,
            kv_footprint=total_footprint - 60.0,
            activation_footprint=0.0,
            weight_traffic=60.0,
            kv_traffic=total_footprint - 60.0,
            activation_traffic=0.0,
        ),
        b_remote_bytes=4.0,
        n_messages=2,
    )


def test_fixed_fraction_is_default_and_matches_explicit_mode_exactly():
    default_parameters = parameters()
    explicit_parameters = replace(
        default_parameters,
        local_hbm_placement_mode="fixed_fraction",
    )

    assert default_parameters.local_hbm_placement_mode == "fixed_fraction"
    assert run_simulation(legacy_signature(), default_parameters) == run_simulation(
        legacy_signature(), explicit_parameters
    )


def test_prefill_capacity_overflow_accepts_nearest_one_ulp_semantic_route_total():
    """Regression: an unrepresentable exact stage sum must not oscillate."""
    signature = build_prefill_signature("Llama-70B", 2, 4096, 16)
    footprint = signature.local_footprint_bytes
    assert footprint == 80_737_418_240.0
    physical_capacity = footprint * 0.16

    # D-1 materialized scratch headroom is one 536,870,912-byte H2H round
    # chunk at TP=2, leaving this hand-checked weight-only placement target.
    placement_capacity = physical_capacity - 536_870_912.0
    assert placement_capacity == 12_381_116_006.4
    streams = split_traffic(
        signature,
        r_remote=0.0,
        r=0.0,
        local_hbm_placement_mode="capacity_overflow",
        total_local_capacity_bytes=placement_capacity,
    )

    stages = streams.semantic_stages
    assert stages is not None
    weight_local = math.fsum(stage.weight_local_bytes for stage in stages)
    kv_local = math.fsum(stage.kv_local_bytes for stage in stages)
    activation_local = math.fsum(
        stage.activation_local_bytes for stage in stages
    )
    stage_total = math.fsum(stage.local_hbm_bytes for stage in stages)
    assert kv_local == 0.0
    assert activation_local == 0.0
    assert abs(stage_total - streams.local_hbm_bytes) <= math.ulp(
        streams.local_hbm_bytes
    )
    assert abs(weight_local - streams.local_hbm_bytes) <= math.ulp(
        streams.local_hbm_bytes
    )

    result = run_simulation(
        signature,
        parameters(
            tp=2,
            r=0.0,
            r_remote=0.0,
            local_hbm_stack_count=4,
            local_hbm_stack_bw_bps=1.2e12,
            local_hbm_stack_capacity_bytes=physical_capacity / 4.0,
            side_hbm_bw_bps=4.8e12,
            h2h_bw_bps=336e9,
            gpu_direct_bw_bps=768e9,
            gpu_peak_flops=990e12,
            base_die_latency_s=5e-9,
            local_hbm_placement_mode="capacity_overflow",
            local_side_memory_concurrency="full_overlap",
            allow_overlap=True,
        ),
    )
    assert result.elapsed_s > 0.0


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("seq_len,batch", [(4096, 16), (16384, 4)])
def test_prefill_weight_only_capacity_boundary_reconciles_for_audited_grid_cases(
    tp,
    seq_len,
    batch,
):
    """The 0.16x weight-only boundary must terminate for every audited TP."""
    signature = build_prefill_signature("Llama-70B", tp, seq_len, batch)
    footprint = signature.local_footprint_bytes
    assert footprint is not None
    assert signature.semantic_stages is not None
    physical_capacity = footprint * 0.16
    rounds = 2 * (tp - 1)
    headroom = max(
        stage.collective_bytes / rounds for stage in signature.semantic_stages
    )
    placement_capacity = max(physical_capacity - headroom, 0.0)

    streams = split_traffic(
        signature,
        r_remote=0.0,
        r=0.0,
        local_hbm_placement_mode="capacity_overflow",
        total_local_capacity_bytes=placement_capacity,
    )

    stages = streams.semantic_stages
    assert stages is not None
    stage_total = math.fsum(stage.local_hbm_bytes for stage in stages)
    assert streams.class_placement is not None
    assert streams.class_placement.kv_local == 0.0
    assert abs(stage_total - streams.local_hbm_bytes) <= math.ulp(
        streams.local_hbm_bytes
    )


def test_reconcile_keeps_existing_bit_exact_correction_result():
    routes = ((10_000_000_000.0, 1.0), (0.0, 0.0), (0.0, 0.0))

    reconciled = workload_module._reconcile_class_route_total(
        routes,
        10_000_000_000.0,
    )

    assert reconciled == (
        (10_000_000_000.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )


def test_capacity_overflow_applies_each_placement_fraction_to_its_own_traffic():
    signature = replace(
        placement_signature(),
        b_remote_bytes=80.0,
        n_messages=1,
    )
    streams = split_traffic(
        signature,
        r_remote=0.99,
        r=0.25,
        local_hbm_placement_mode="capacity_overflow",
        total_local_capacity_bytes=60.0,
    )
    ignored_r_remote = split_traffic(
        signature,
        r_remote=0.0,
        r=0.25,
        local_hbm_placement_mode="capacity_overflow",
        total_local_capacity_bytes=60.0,
    )

    # Placement is weight=100% local, KV=50% local, activation=0% local.
    assert streams.local_hbm_bytes == 105.0
    assert streams.side_hbm_bytes == 1_005.0
    assert streams.local_resident_footprint_bytes == 60.0
    assert streams.side_resident_footprint_bytes == 40.0
    assert streams.effective_r_remote == 1_005.0 / 1_110.0
    assert streams.h2h_bytes == 60.0
    assert streams.gpu_direct_bytes == 20.0
    assert streams == ignored_r_remote

    # A wrong aggregate-overflow split would produce 666/444 bytes here.
    assert streams.local_hbm_bytes != 666.0
    assert streams.side_hbm_bytes != 444.0


def test_capacity_overflow_checks_post_placement_local_resident_footprint():
    signature = placement_signature(
        weight_footprint=60e9,
        kv_footprint=20e9,
        activation_footprint=0.0,
        weight_traffic=60e9,
        kv_traffic=20e9,
        activation_traffic=0.0,
    )

    # Total footprint is 80 GB, but placement leaves exactly 64 GB local. This
    # must not be rejected by comparing the pre-placement total to local capacity.
    result = run_simulation(
        signature,
        parameters(
            r_remote=0.0,
            local_hbm_placement_mode="capacity_overflow",
            local_hbm_stack_count=4,
            local_hbm_stack_bw_bps=1e12,
            local_hbm_stack_capacity_bytes=16e9,
            side_hbm_bw_bps=1e12,
        ),
    )

    assert result.elapsed_s == 0.016


def test_materialized_headroom_makes_partial_spill_feasible():
    signature = materialized_spill_signature()
    params = parameters(
        tp=4,
        local_hbm_placement_mode="capacity_overflow",
        local_hbm_stack_count=4,
        local_hbm_stack_capacity_bytes=16.0,
    )

    result = run_simulation(signature, params)
    streams = split_traffic(
        signature,
        r_remote=0.25,
        r=0.0,
        local_hbm_placement_mode="capacity_overflow",
        total_local_capacity_bytes=64.0 - (4.0 / 2.0 / 6.0),
    )

    # TP=4 has six ring rounds. H_materialized is the reusable per-round
    # scratch chunk: (4 / 2) / 6 = 1/3 byte, not a full collective payload.
    expected_spill = 16.0 + (1.0 / 3.0)
    assert result.resource_metrics["side_hbm[0]"].transferred_amount == pytest.approx(
        expected_spill
    )
    assert streams.effective_r_remote == pytest.approx(expected_spill / 80.0)
    assert streams.local_resident_footprint_bytes + (1.0 / 3.0) <= 64.0 + 1e-12


def test_materialized_headroom_uses_weighted_mapping_capacity():
    signature = materialized_spill_signature()
    params = parameters(
        tp=4,
        local_hbm_placement_mode="capacity_overflow",
        local_hbm_stack_count=4,
        local_hbm_stack_capacity_bytes=16.0,
        local_hbm_mapping_policy="weighted",
        local_hbm_mapping_weights=(0.5, 0.25, 0.125, 0.125),
    )

    result = run_simulation(signature, params)

    # The hottest stack binds C_mapped at 32 bytes. Reserving the TP=4
    # per-round scratch chunk (1/3 byte) leaves 31 2/3 resident bytes.
    expected_spill = 48.0 + (1.0 / 3.0)
    assert result.resource_metrics["side_hbm[0]"].transferred_amount == pytest.approx(
        expected_spill
    )
    assert ((32.0 - (1.0 / 3.0)) + (1.0 / 3.0)) * 0.5 == 16.0


def test_uniform_mapping_uses_physical_capacity_without_division_roundoff():
    stack_capacity = 1.3709598073165986e19
    physical_capacity = 3 * stack_capacity
    signature = placement_signature(
        weight_footprint=physical_capacity + 1e12,
        kv_footprint=0.0,
        activation_footprint=0.0,
        weight_traffic=physical_capacity + 1e12,
        kv_traffic=0.0,
        activation_traffic=0.0,
    )

    result = run_simulation(
        signature,
        parameters(
            tp=3,
            comm_mode="cut_through",
            local_hbm_placement_mode="capacity_overflow",
            local_hbm_stack_count=3,
            local_hbm_stack_capacity_bytes=stack_capacity,
            local_hbm_stack_bw_bps=1e30,
            side_hbm_bw_bps=1e30,
        ),
    )

    assert result.resource_metrics["side_hbm[0]"].transferred_amount > 0.0


def test_materialized_exact_fit_tolerates_subtract_add_roundoff():
    signature = placement_signature(
        weight_footprint=1e-5,
        kv_footprint=0.0,
        activation_footprint=0.0,
        weight_traffic=1e-5,
        kv_traffic=0.0,
        activation_traffic=0.0,
    )
    signature = replace(
        signature,
        b_remote_bytes=1.0000000000000002e-6,
        n_messages=1,
    )

    # IEEE-754 gives (5e-6 - H) + H one ULP above 5e-6 for this H. The
    # mathematically exact-fit placement must not fail the downstream check.
    result = run_simulation(
        signature,
        parameters(
            tp=2,
            local_hbm_placement_mode="capacity_overflow",
            local_hbm_stack_capacity_bytes=5e-6,
            local_hbm_stack_bw_bps=1e6,
            side_hbm_bw_bps=1e6,
            h2h_bw_bps=1e6,
        ),
    )

    assert result.resource_metrics["side_hbm[0]"].transferred_amount > 0.0


def test_materialized_exact_fit_tolerates_class_boundary_roundoff():
    stack_capacity = 1_276_831_220.5911963
    headroom = 364_512_452.7595693
    signature = placement_signature(
        weight_footprint=47_440_575.9272446,
        kv_footprint=1_000_000_000.0,
        activation_footprint=0.0,
        weight_traffic=47_440_575.9272446,
        kv_traffic=1_000_000_000.0,
        activation_traffic=0.0,
    )
    signature = replace(
        signature,
        # TP=2 has two rounds, so aggregate H2H bytes must be twice the
        # intended per-round scratch headroom used by this numeric fixture.
        b_remote_bytes=headroom * 2.0,
        n_messages=1,
    )

    # Placement crosses from weight into KV. Subtracting the weight from
    # C_usable and adding both class allocations reconstructs resident bytes
    # one ULP above C_usable, even though the mathematical fit is exact.
    result = run_simulation(
        signature,
        parameters(
            tp=2,
            local_hbm_placement_mode="capacity_overflow",
            local_hbm_stack_capacity_bytes=stack_capacity,
            local_hbm_stack_bw_bps=1e15,
            side_hbm_bw_bps=1e15,
            h2h_bw_bps=1e15,
        ),
    )

    assert result.resource_metrics["side_hbm[0]"].transferred_amount > 0.0


def test_cut_through_capacity_overflow_keeps_full_physical_capacity():
    result = run_simulation(
        materialized_spill_signature(),
        parameters(
            tp=4,
            comm_mode="cut_through",
            local_hbm_placement_mode="capacity_overflow",
            local_hbm_stack_count=4,
            local_hbm_stack_capacity_bytes=16.0,
        ),
    )

    # Cut-through has no materialized receive buffer, so the original full
    # 64-byte pooled capacity remains available and only 16 bytes spill.
    assert result.resource_metrics["side_hbm[0]"].transferred_amount == 16.0


@pytest.mark.parametrize(
    ("r", "include_endpoints", "expected_side_bytes"),
    [
        (0.5, True, 16.0 + (1.0 / 6.0)),
        (1.0, True, 16.0),
        (0.0, False, 16.0),
    ],
)
def test_materialized_headroom_uses_only_endpoint_h2h_bytes(
    r, include_endpoints, expected_side_bytes
):
    result = run_simulation(
        materialized_spill_signature(),
        parameters(
            tp=4,
            r=r,
            include_materialized_endpoints=include_endpoints,
            local_hbm_placement_mode="capacity_overflow",
            local_hbm_stack_count=4,
            local_hbm_stack_capacity_bytes=16.0,
        ),
    )

    assert result.resource_metrics["side_hbm[0]"].transferred_amount == pytest.approx(
        expected_side_bytes
    )


def test_weighted_cut_through_uses_mapping_limited_capacity_without_headroom():
    result = run_simulation(
        materialized_spill_signature(),
        parameters(
            tp=4,
            comm_mode="cut_through",
            local_hbm_placement_mode="capacity_overflow",
            local_hbm_stack_count=4,
            local_hbm_stack_capacity_bytes=16.0,
            local_hbm_mapping_policy="weighted",
            local_hbm_mapping_weights=(0.5, 0.25, 0.125, 0.125),
        ),
    )

    # C_mapped = 16 / 0.5 = 32 and cut-through reserves no buffer.
    assert result.resource_metrics["side_hbm[0]"].transferred_amount == 48.0


def test_materialized_buffer_larger_than_mapped_capacity_remains_infeasible():
    with pytest.raises(ValueError, match="materialized receive buffer"):
        run_simulation(
            materialized_spill_signature(),
            parameters(
                tp=4,
                local_hbm_placement_mode="capacity_overflow",
                local_hbm_stack_capacity_bytes=0.05,
            ),
        )


def test_materialized_headroom_does_not_spill_when_footprint_plus_buffer_fits():
    result = run_simulation(
        materialized_spill_signature(total_footprint=60.0),
        parameters(
            tp=4,
            local_hbm_placement_mode="capacity_overflow",
            local_hbm_stack_count=4,
            local_hbm_stack_capacity_bytes=16.0,
        ),
    )

    # 60 resident bytes + 2 buffer bytes fit in 64 physical bytes.
    assert result.resource_metrics["side_hbm[0]"].transferred_amount == 0.0


def test_capacity_overflow_can_place_only_part_of_weight_locally():
    streams = split_traffic(
        placement_signature(
            weight_footprint=80.0,
            kv_footprint=10.0,
            activation_footprint=10.0,
            weight_traffic=20.0,
            kv_traffic=100.0,
            activation_traffic=1_000.0,
        ),
        r_remote=0.0,
        r=0.0,
        local_hbm_placement_mode="capacity_overflow",
        total_local_capacity_bytes=40.0,
    )

    assert streams.local_resident_footprint_bytes == 40.0
    assert streams.side_resident_footprint_bytes == 60.0
    assert streams.local_hbm_bytes == 10.0
    assert streams.side_hbm_bytes == 1_110.0


def test_capacity_overflow_enforces_side_capacity_on_resident_bytes():
    # Resident overflow is 40 bytes while remote traffic is 1,005 bytes. A
    # 100-byte side capacity must pass if capacity checks resident state only.
    streams = split_traffic(
        placement_signature(),
        r_remote=0.0,
        r=0.0,
        local_hbm_placement_mode="capacity_overflow",
        total_local_capacity_bytes=60.0,
        side_hbm_capacity_bytes=100.0,
    )
    assert streams.side_resident_footprint_bytes == 40.0
    assert streams.side_hbm_bytes == 1_005.0

    # Conversely, zero traffic cannot hide a resident-capacity overflow.
    with pytest.raises(workload_module.SideHBMCapacityError):
        split_traffic(
            placement_signature(
                weight_traffic=0.0,
                kv_traffic=0.0,
                activation_traffic=0.0,
            ),
            r_remote=0.0,
            r=0.0,
            local_hbm_placement_mode="capacity_overflow",
            total_local_capacity_bytes=60.0,
            side_hbm_capacity_bytes=39.0,
        )


def test_prefill_activation_footprint_and_traffic_remain_independent():
    signature = build_prefill_signature(
        "Llama-7B",
        tp=2,
        seq_len=16,
        batch=2,
        c_act_footprint=2.0,
        c_act_traffic=0.0,
    )

    assert signature.weight_footprint_bytes == 7_000_000_000.0
    assert signature.kv_footprint_bytes == 8_388_608.0
    assert signature.activation_footprint_bytes == 524_288.0
    assert signature.activation_traffic_bytes_value == 0.0
    assert signature.local_footprint_bytes == 7_008_912_896.0
    assert signature.b_local_bytes == 7_008_388_608.0

    streams = split_traffic(
        signature,
        r_remote=0.5,
        r=0.0,
        local_hbm_placement_mode="capacity_overflow",
        total_local_capacity_bytes=7_008_388_608.0,
    )
    assert streams.side_resident_footprint_bytes == 524_288.0
    assert streams.side_hbm_bytes == 0.0
    assert streams.effective_r_remote == 0.0

    capacity_signature = replace(
        signature,
        b_remote_bytes=0.0,
        n_messages=0,
        compute_flops=0.0,
    )
    with pytest.raises(ValueError):
        run_simulation(
            capacity_signature,
            parameters(
                tp=2,
                r_remote=0.0,
                local_hbm_stack_bw_bps=1e12,
                local_hbm_stack_capacity_bytes=7_008_388_608.0,
                side_hbm_bw_bps=1e12,
            ),
        )


def test_builders_populate_class_fields_and_default_prefill_footprint_exactly():
    decode = build_decode_signature("Llama-70B", tp=4, seq_len=4096, batch=1)
    prefill = build_prefill_signature("Llama-70B", tp=4, seq_len=4096, batch=1)

    assert decode.weight_footprint_bytes == decode.weight_traffic_bytes == 35e9
    assert decode.kv_footprint_bytes == decode.kv_traffic_bytes == 335_544_320.0
    assert decode.activation_footprint_bytes == 0.0
    assert decode.activation_traffic_bytes_value == 0.0
    assert decode.local_footprint_bytes == 35_335_544_320.0

    assert prefill.activation_footprint_bytes == 0.0
    assert prefill.local_footprint_bytes == (
        prefill.weight_footprint_bytes
        + prefill.kv_footprint_bytes
        + prefill.activation_footprint_bytes
    )
    assert prefill.local_footprint_bytes == 35_335_544_320.0


def test_capacity_overflow_requires_class_decomposition_and_known_mode():
    with pytest.raises(ValueError, match="class-level footprint and traffic"):
        split_traffic(
            legacy_signature(),
            r_remote=0.0,
            r=0.0,
            local_hbm_placement_mode="capacity_overflow",
            total_local_capacity_bytes=20.0,
        )

    with pytest.raises(ValueError, match="local_hbm_placement_mode"):
        split_traffic(
            legacy_signature(),
            r_remote=0.0,
            r=0.0,
            local_hbm_placement_mode="unknown",
        )


def test_fixed_fraction_exposes_traffic_fraction_and_resident_placement():
    streams = split_traffic(
        legacy_signature(),
        r_remote=0.25,
        r=0.0,
        local_hbm_placement_mode="fixed_fraction",
    )

    assert streams.local_hbm_bytes == 75.0
    assert streams.side_hbm_bytes == 25.0
    assert streams.effective_r_remote == 0.25
    assert streams.local_resident_footprint_bytes == 30.0
    assert streams.side_resident_footprint_bytes == 10.0


def test_fixed_fraction_preserves_side_request_count_for_zero_local_traffic():
    signature = TrafficSignature(
        phase="decode",
        b_local_bytes=0.0,
        b_remote_bytes=0.0,
        n_messages=0,
        compute_flops=0.0,
        n_side_requests=7,
    )

    streams = split_traffic(signature, r_remote=0.5, r=0.0)

    # Preserve the pre-capacity-overflow fixed-fraction condition exactly:
    # request count follows r_remote, even for a zero-byte synthetic signature.
    assert streams.n_side_requests == 7
