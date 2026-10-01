import stage1_model

from stage1_model import MODELS, b_local_per_token
from uhbm_simulator.simulator import SimulationParameters, run_simulation
from uhbm_simulator.workload import build_decode_signature, build_prefill_signature


def test_prefill_signature_is_compute_dominant_and_scales_collective_payload():
    signature = build_prefill_signature("Llama-70B", tp=4, seq_len=4096, batch=1)

    assert signature.phase == "prefill"
    assert signature.token_count == 4096
    assert signature.compute_flops == 154_355_116_277_760.0
    assert signature.b_remote_bytes > 1e9


def test_prefill_attention_flops_match_dense_formula_and_causal_half_reference():
    model = {"n_layers": 2, "d_model": 8}

    dense = stage1_model.attention_flops_prefill(
        model, tp=1, seq_len=4, batch=3, c_causal=1.0
    )
    triangular = stage1_model.attention_flops_prefill(
        model, tp=1, seq_len=4, batch=3, c_causal=0.5
    )

    assert dense == 3_072.0
    assert triangular == 1_536.0


def test_prefill_attention_is_quadratic_while_linear_work_is_linear_in_sequence():
    model = MODELS["Llama-70B"]
    short = build_prefill_signature("Llama-70B", tp=4, seq_len=8, batch=2)
    long = build_prefill_signature("Llama-70B", tp=4, seq_len=16, batch=2)
    short_attention = stage1_model.attention_flops_prefill(
        model, tp=4, seq_len=8, batch=2
    )
    long_attention = stage1_model.attention_flops_prefill(
        model, tp=4, seq_len=16, batch=2
    )

    assert long_attention == 4.0 * short_attention
    assert long.compute_flops - long_attention == 2.0 * (
        short.compute_flops - short_attention
    )


def test_activation_footprint_and_traffic_scale_independently_without_tp_sharding():
    model = {"d_model": 8}

    footprint_only = stage1_model.activation_footprint_bytes(
        model, tp=4, seq_len=4, batch=3, c_act_footprint=2.0
    )
    no_traffic = stage1_model.activation_traffic_bytes(
        model, tp=4, seq_len=4, batch=3, c_act_traffic=0.0
    )
    no_footprint = stage1_model.activation_footprint_bytes(
        model, tp=1, seq_len=4, batch=3, c_act_footprint=0.0
    )
    traffic_only_tp_one = stage1_model.activation_traffic_bytes(
        model, tp=1, seq_len=4, batch=3, c_act_traffic=3.0
    )
    traffic_only_tp_four = stage1_model.activation_traffic_bytes(
        model, tp=4, seq_len=4, batch=3, c_act_traffic=3.0
    )

    assert footprint_only == 384.0
    assert no_traffic == 0.0
    assert no_footprint == 0.0
    assert traffic_only_tp_one == 576.0
    assert traffic_only_tp_four == traffic_only_tp_one


def test_zero_activation_coefficients_preserve_prefill_local_traffic_and_footprint():
    model = MODELS["Llama-70B"]
    signature = build_prefill_signature(
        "Llama-70B", tp=4, seq_len=4096, batch=1, c_act_traffic=0.0
    )
    original_local = b_local_per_token(model, 4, 4096, 1)

    assert signature.b_local_bytes == original_local
    assert signature.local_footprint_bytes == original_local
    assert (
        stage1_model.activation_footprint_bytes(
            model, tp=4, seq_len=4096, batch=1, c_act_footprint=0.0
        )
        == 0.0
    )


def test_prefill_signature_combines_linear_attention_and_activation_traffic():
    signature = build_prefill_signature(
        "Llama-70B",
        tp=4,
        seq_len=16,
        batch=2,
        c_causal=0.5,
        c_act_traffic=3.0,
    )

    assert signature.compute_flops == 1_120_167_772_160.0
    assert signature.b_local_bytes == 35_004_194_304.0
    assert signature.local_footprint_bytes == 35_002_621_440.0


def test_decode_signature_remains_unchanged_by_prefill_only_coefficients():
    signature = build_decode_signature("Llama-70B", tp=4, seq_len=4096, batch=2)

    assert signature.b_local_bytes == 35_671_088_640.0
    assert signature.local_footprint_bytes == signature.b_local_bytes
    assert signature.b_remote_bytes == 7_864_320.0
    assert signature.n_messages == 160
    assert signature.compute_flops == 70_000_000_000.0
    assert signature.token_count == 1
    assert signature.semantic_stages is not None
    assert len(signature.semantic_stages) == 160


def test_prefill_wrapper_inherits_llama2_70b_gqa_aware_local_bytes():
    signature = build_prefill_signature("Llama-70B", tp=4, seq_len=4096, batch=1)

    assert signature.b_local_bytes == 35_335_544_320.0


def test_prefill_reports_ttft_and_amortized_tpot():
    signature = build_prefill_signature("Llama-7B", tp=1, seq_len=10, batch=1)
    parameters = SimulationParameters(
        tp=1,
        r=0.0,
        r_remote=0.0,
        comm_mode="materialized",
        thermal_scenario="no_throttling",
        local_hbm_stack_count=1,
        local_hbm_stack_bw_bps=1e30,
        local_hbm_stack_capacity_bytes=1e30,
        local_hbm_mapping_policy="uniform",
        local_hbm_mapping_weights=None,
        side_hbm_bw_bps=1e30,
        side_hbm_latency_s=0.0,
        h2h_bw_bps=1e30,
        h2h_latency_s=0.0,
        gpu_direct_bw_bps=1e30,
        gpu_direct_latency_s=0.0,
        gpu_peak_flops=1e12,
        base_die_latency_s=0.0,
    )

    result = run_simulation(signature, parameters)

    assert result.ttft_s == result.tpot_s * 10
