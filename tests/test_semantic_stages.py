from dataclasses import replace
import math

import pytest

from stage1_model import MODELS, attention_parameter_fraction
from uhbm_simulator.simulator import SimulationParameters, run_simulation
from uhbm_simulator.workload import (
    StageSignature,
    TrafficSignature,
    build_decode_signature,
    build_prefill_signature,
    split_traffic,
)


def _parameters(**overrides):
    base = SimulationParameters(
        tp=2,
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
        # Keep this D-2 stage-order fixture independent of D-1 side transport.
        side_hbm_path="independent",
    )
    return replace(base, **overrides)


def test_llama2_parameter_fractions_use_gqa_and_intermediate_size():
    assert MODELS["Llama-7B"]["intermediate_size"] == 11008
    assert MODELS["Llama-70B"]["intermediate_size"] == 28672

    # Llama-2-70B: P_attn=150,994,944 and P_mlp=704,643,072 per layer.
    assert attention_parameter_fraction(MODELS["Llama-70B"]) == 3.0 / 17.0
    assert attention_parameter_fraction(
        MODELS["Llama-7B"]
    ) == pytest.approx(0.3316062176165803, rel=0.0, abs=1e-15)


def test_decode_semantic_stages_preserve_aggregate_traffic_compute_and_collectives():
    signature = build_decode_signature("Llama-70B", tp=4, seq_len=4096, batch=8)
    stages = signature.semantic_stages

    assert stages is not None
    assert len(stages) == 160
    assert all(
        stage.layer_index == index // 2
        and stage.sublayer == ("attention" if index % 2 == 0 else "mlp")
        for index, stage in enumerate(stages)
    )
    assert math.fsum(stage.weight_traffic_bytes for stage in stages) == (
        signature.weight_traffic_bytes
    )
    assert math.fsum(stage.kv_traffic_bytes for stage in stages) == (
        signature.kv_traffic_bytes
    )
    assert math.fsum(stage.activation_traffic_bytes for stage in stages) == 0.0
    assert math.fsum(stage.compute_flops for stage in stages) == (
        signature.compute_flops
    )
    assert math.fsum(stage.collective_bytes for stage in stages) == (
        signature.b_remote_bytes
    )
    assert math.fsum(stage.weight_traffic_bytes for stage in stages[::2]) == (
        signature.weight_traffic_bytes * (3.0 / 17.0)
    )
    assert math.fsum(stage.weight_traffic_bytes for stage in stages[1::2]) == (
        signature.weight_traffic_bytes
        - signature.weight_traffic_bytes * (3.0 / 17.0)
    )
    assert math.fsum(stage.compute_flops for stage in stages[::2]) == (
        signature.compute_flops * (3.0 / 17.0)
    )
    assert math.fsum(stage.compute_flops for stage in stages[1::2]) == (
        signature.compute_flops
        - signature.compute_flops * (3.0 / 17.0)
    )
    assert all(stage.kv_traffic_bytes > 0.0 for stage in stages[::2])
    assert all(stage.kv_traffic_bytes == 0.0 for stage in stages[1::2])
    assert all(stage.weight_traffic_bytes > 0.0 for stage in stages)


def test_prefill_semantic_stages_keep_quadratic_compute_in_attention_and_spread_activation():
    signature = build_prefill_signature(
        "Llama-70B",
        tp=4,
        seq_len=16,
        batch=2,
        c_causal=0.5,
        c_act_traffic=3.0,
    )
    stages = signature.semantic_stages

    assert stages is not None
    assert math.fsum(stage.activation_traffic_bytes for stage in stages) == (
        signature.activation_traffic_bytes_value
    )
    expected_activation_per_stage = (
        signature.activation_traffic_bytes_value / len(stages)
    )
    assert all(
        stage.activation_traffic_bytes
        == pytest.approx(expected_activation_per_stage, rel=0.0, abs=1e-9)
        for stage in stages
    )
    assert math.fsum(stage.compute_flops for stage in stages) == (
        signature.compute_flops
    )

    linear_flops = 2.0 * 70e9 * 16 * 2 / 4
    quadratic_flops = 0.5 * 4 * 80 * 2 * 16**2 * 8192 / 4
    assert math.fsum(stage.compute_flops for stage in stages[::2]) == pytest.approx(
        linear_flops * (3.0 / 17.0) + quadratic_flops,
        rel=0.0,
        abs=1e-4,
    )
    assert math.fsum(stage.compute_flops for stage in stages[1::2]) == pytest.approx(
        linear_flops * (14.0 / 17.0),
        rel=0.0,
        abs=1e-4,
    )


def test_fixed_fraction_r_remote_doubles_side_requests_under_semantic_stages():
    """Remote weights make both sub-layers active; 2L is intentional."""
    signature = build_decode_signature("Llama-70B", tp=4, seq_len=4096)

    streams = split_traffic(signature, r_remote=0.1, r=0.0)

    assert streams.n_side_requests == 2 * MODELS["Llama-70B"]["n_layers"]
    assert streams.n_side_requests != MODELS["Llama-70B"]["n_layers"]
    assert streams.semantic_stages is not None
    assert all(stage.side_hbm_bytes > 0.0 for stage in streams.semantic_stages)


def test_capacity_overflow_stage_split_applies_each_class_placement_fraction():
    stages = (
        StageSignature(0, "attention", 30.0, 20.0, 0.0, 30.0, 10.0),
        StageSignature(0, "mlp", 70.0, 0.0, 0.0, 70.0, 10.0),
    )
    signature = TrafficSignature(
        phase="decode",
        b_local_bytes=120.0,
        b_remote_bytes=20.0,
        n_messages=2,
        compute_flops=100.0,
        local_footprint_bytes=120.0,
        weight_footprint_bytes=100.0,
        kv_footprint_bytes=20.0,
        activation_footprint_bytes=0.0,
        weight_traffic_bytes=100.0,
        kv_traffic_bytes=20.0,
        activation_traffic_bytes_value=0.0,
        semantic_stages=stages,
    )

    streams = split_traffic(
        signature,
        r_remote=0.99,
        r=0.25,
        local_hbm_placement_mode="capacity_overflow",
        total_local_capacity_bytes=110.0,
    )

    assert streams.class_placement.weight_local == 1.0
    assert streams.class_placement.kv_local == 0.5
    assert streams.class_placement.activation_local == 1.0
    assert streams.semantic_stages is not None
    attention, mlp = streams.semantic_stages
    assert attention.weight_local_bytes == 30.0
    assert attention.weight_side_bytes == 0.0
    assert attention.kv_local_bytes == 10.0
    assert attention.kv_side_bytes == 10.0
    assert mlp.kv_local_bytes == mlp.kv_side_bytes == 0.0
    assert math.fsum(stage.local_hbm_bytes for stage in streams.semantic_stages) == (
        streams.local_hbm_bytes
    )
    assert math.fsum(stage.side_hbm_bytes for stage in streams.semantic_stages) == (
        streams.side_hbm_bytes
    )
    assert math.fsum(stage.h2h_bytes for stage in streams.semantic_stages) == (
        streams.h2h_bytes
    )
    assert math.fsum(stage.gpu_direct_bytes for stage in streams.semantic_stages) == (
        streams.gpu_direct_bytes
    )


def test_capacity_overflow_remote_kv_activates_only_attention_side_requests():
    signature = build_decode_signature("Llama-70B", tp=4, seq_len=4096)

    streams = split_traffic(
        signature,
        r_remote=0.0,
        r=0.0,
        local_hbm_placement_mode="capacity_overflow",
        total_local_capacity_bytes=signature.weight_footprint_bytes,
    )

    assert streams.semantic_stages is not None
    assert streams.class_placement.weight_local == 1.0
    assert streams.class_placement.kv_local == 0.0
    assert streams.n_side_requests == MODELS["Llama-70B"]["n_layers"]
    assert all(stage.side_hbm_bytes > 0.0 for stage in streams.semantic_stages[::2])
    assert all(stage.side_hbm_bytes == 0.0 for stage in streams.semantic_stages[1::2])


@pytest.mark.parametrize("fraction", [0.1, 1e-20])
def test_semantic_routes_conserve_both_complementary_aggregates_exactly(fraction):
    """Do not reconstruct a small route as total-minus-rounded-large-route."""
    signature = build_decode_signature("Llama-70B", tp=4, seq_len=4096)

    streams = split_traffic(signature, r_remote=fraction, r=fraction)

    assert streams.semantic_stages is not None
    assert math.fsum(
        stage.local_hbm_bytes for stage in streams.semantic_stages
    ) == streams.local_hbm_bytes
    assert math.fsum(
        stage.side_hbm_bytes for stage in streams.semantic_stages
    ) == streams.side_hbm_bytes
    assert math.fsum(stage.h2h_bytes for stage in streams.semantic_stages) == (
        streams.h2h_bytes
    )
    assert math.fsum(
        stage.gpu_direct_bytes for stage in streams.semantic_stages
    ) == streams.gpu_direct_bytes
    assert streams.n_side_requests == 2 * MODELS["Llama-70B"]["n_layers"]


def test_semantic_route_conservation_covers_other_models_and_activation_traffic():
    signatures = (
        build_decode_signature("Llama-7B", tp=2, seq_len=4096),
        build_prefill_signature(
            "Llama-70B",
            tp=4,
            seq_len=128,
            batch=2,
            c_act_traffic=3.0,
        ),
    )

    for signature in signatures:
        streams = split_traffic(signature, r_remote=0.1, r=0.1)
        assert streams.semantic_stages is not None
        assert math.fsum(
            stage.local_hbm_bytes for stage in streams.semantic_stages
        ) == streams.local_hbm_bytes
        assert math.fsum(
            stage.side_hbm_bytes for stage in streams.semantic_stages
        ) == streams.side_hbm_bytes
        assert math.fsum(
            stage.h2h_bytes for stage in streams.semantic_stages
        ) == streams.h2h_bytes
        assert math.fsum(
            stage.gpu_direct_bytes for stage in streams.semantic_stages
        ) == streams.gpu_direct_bytes


def test_semantic_strict_execution_uses_stage_specific_memory_and_collective_order():
    stages = (
        StageSignature(0, "attention", 100.0, 100.0, 0.0, 100.0, 100.0),
        StageSignature(0, "mlp", 100.0, 0.0, 0.0, 100.0, 100.0),
    )
    signature = TrafficSignature(
        phase="decode",
        b_local_bytes=300.0,
        b_remote_bytes=200.0,
        n_messages=2,
        compute_flops=200.0,
        local_footprint_bytes=300.0,
        weight_footprint_bytes=200.0,
        kv_footprint_bytes=100.0,
        activation_footprint_bytes=0.0,
        weight_traffic_bytes=200.0,
        kv_traffic_bytes=100.0,
        activation_traffic_bytes_value=0.0,
        semantic_stages=stages,
    )

    full_overlap = run_simulation(signature, _parameters())
    serialized = run_simulation(
        signature,
        _parameters(local_side_memory_concurrency="serialized"),
    )

    assert full_overlap.elapsed_s == 5.0
    assert serialized.elapsed_s == 6.0
    assert full_overlap.resource_metrics["side_hbm[0]"].request_count == 2
    assert serialized.resource_metrics["side_hbm[0]"].request_count == 2
