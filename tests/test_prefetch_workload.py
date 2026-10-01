import math

import pytest

from stage1_model import MODELS, b_kv_footprint, b_weight_footprint


def test_decode_layer_plans_conserve_weight_historical_kv_and_new_kv():
    """Dropping or duplicating bytes while decomposing layers must fail."""
    from uhbm_simulator.prefetch_workload import build_layer_plans

    model = MODELS["Llama-70B"]
    plans = build_layer_plans(
        "Llama-70B",
        tp=4,
        seq_len=4096,
        batch=8,
        phase="decode",
        decode_tokens=3,
        token_index=1,
    )

    assert len(plans) == model["n_layers"]
    assert math.fsum(plan.weight_bytes for plan in plans) == b_weight_footprint(
        model, 4
    )
    assert math.fsum(plan.historical_kv_bytes for plan in plans) == b_kv_footprint(
        model, 4, 4097, 8
    )
    assert math.fsum(plan.new_kv_bytes for plan in plans) == (
        b_kv_footprint(model, 4, 4098, 8)
        - b_kv_footprint(model, 4, 4097, 8)
    )


def test_uniform_lane_mapping_conserves_each_layer_with_residual_final_lane():
    """A naive rounded 1/8 split that loses the residual must fail."""
    from uhbm_simulator.prefetch_workload import build_layer_plans

    plan = build_layer_plans(
        "Llama-70B",
        tp=4,
        seq_len=4097,
        batch=3,
        phase="decode",
    )[0]

    assert len(plan.lanes) == 8
    assert tuple(lane.lane_index for lane in plan.lanes) == tuple(range(8))
    assert math.fsum(lane.weight_bytes for lane in plan.lanes) == plan.weight_bytes
    assert math.fsum(lane.historical_kv_bytes for lane in plan.lanes) == (
        plan.historical_kv_bytes
    )
    assert math.fsum(lane.new_kv_bytes for lane in plan.lanes) == plan.new_kv_bytes
    for lane in plan.lanes:
        assert lane.weight_bytes == pytest.approx(plan.weight_bytes / 8.0)
        assert lane.historical_kv_bytes == pytest.approx(
            plan.historical_kv_bytes / 8.0
        )
        assert lane.new_kv_bytes == pytest.approx(plan.new_kv_bytes / 8.0)


def test_fresh_prefill_has_no_historical_kv_prefetch_but_generates_prompt_kv():
    """Treating prompt KV as historical prefetch traffic must fail."""
    from uhbm_simulator.prefetch_workload import build_layer_plans

    model = MODELS["Llama-7B"]
    plans = build_layer_plans(
        "Llama-7B",
        tp=4,
        seq_len=2048,
        batch=2,
        phase="prefill",
    )

    assert all(plan.historical_kv_bytes == 0.0 for plan in plans)
    assert math.fsum(plan.new_kv_bytes for plan in plans) == b_kv_footprint(
        model, 4, 2048, 2
    )


def test_decode_historical_kv_grows_by_exactly_one_token_per_step():
    """Fail if token state does not advance KV residency one token at a time."""
    from uhbm_simulator.prefetch_workload import build_layer_plans

    first = build_layer_plans(
        "Llama-70B",
        4,
        4096,
        8,
        phase="decode",
        decode_tokens=2,
        token_index=0,
    )
    second = build_layer_plans(
        "Llama-70B",
        4,
        4096,
        8,
        phase="decode",
        decode_tokens=2,
        token_index=1,
    )

    first_total = math.fsum(plan.historical_kv_bytes for plan in first)
    second_total = math.fsum(plan.historical_kv_bytes for plan in second)
    generated = math.fsum(plan.new_kv_bytes for plan in first)
    assert second_total - first_total == generated


@pytest.mark.parametrize("decode_tokens", [0, -1])
def test_decode_tokens_must_be_positive(decode_tokens):
    """Zero-token decode sweeps must be rejected instead of silently empty."""
    from uhbm_simulator.prefetch_workload import build_layer_plans

    with pytest.raises(ValueError, match="decode_tokens.*positive"):
        build_layer_plans(
            "Llama-70B",
            4,
            4096,
            1,
            phase="decode",
            decode_tokens=decode_tokens,
        )


def test_layer_plans_preserve_d2_attention_mlp_attribution():
    """Merging attention and MLP work would break semantic-layer scheduling."""
    from uhbm_simulator.prefetch_workload import build_layer_plans
    from uhbm_simulator.workload import build_decode_signature

    plans = build_layer_plans(
        "Llama-70B", 4, 4096, 1, phase="decode"
    )
    signature = build_decode_signature("Llama-70B", 4, 4096, 1)

    assert math.fsum(plan.attention_compute_flops for plan in plans) == math.fsum(
        stage.compute_flops for stage in signature.semantic_stages[::2]
    )
    assert math.fsum(plan.mlp_compute_flops for plan in plans) == math.fsum(
        stage.compute_flops for stage in signature.semantic_stages[1::2]
    )
    assert math.fsum(plan.attention_collective_bytes for plan in plans) == math.fsum(
        stage.collective_bytes for stage in signature.semantic_stages[::2]
    )
    assert math.fsum(plan.mlp_collective_bytes for plan in plans) == math.fsum(
        stage.collective_bytes for stage in signature.semantic_stages[1::2]
    )


def test_split_residual_conserves_nonbinary_total_exactly():
    """Equal-share rounding without a residual-corrected last part must fail."""
    from uhbm_simulator.prefetch_workload import split_residual

    parts = split_residual(10.1, 8)
    assert len(parts) == 8
    assert parts[:7] == (10.1 / 8.0,) * 7
    assert math.fsum(parts) == 10.1
