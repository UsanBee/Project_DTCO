import pytest
import stage1_model

from stage1_model import (
    MODELS,
    b_local_per_token,
    b_remote_per_token,
)
from uhbm_simulator.workload import build_decode_signature, split_traffic


def test_decode_signature_reuses_stage1_traffic_functions():
    signature = build_decode_signature("Llama-70B", tp=4, seq_len=4096, batch=1)
    expected_local = b_local_per_token(MODELS["Llama-70B"], 4, 4096, 1)
    expected_remote, expected_messages = b_remote_per_token(
        MODELS["Llama-70B"], 4, 1
    )

    assert signature.b_local_bytes == expected_local
    assert signature.b_local_bytes == 35_335_544_320.0
    assert signature.b_remote_bytes == expected_remote
    assert signature.n_messages == expected_messages
    assert signature.semantic_stages is not None
    assert len(signature.semantic_stages) == 2 * MODELS["Llama-70B"]["n_layers"]


def test_weight_and_kv_footprints_recompose_combined_local_footprint_exactly():
    cases = [
        ("Llama-7B", 2, 128, 1, 7_000_000_000.0, 33_554_432.0),
        ("Llama-70B", 4, 4096, 8, 35_000_000_000.0, 2_684_354_560.0),
        ("Llama-70B", 8, 2048, 4, 17_500_000_000.0, 335_544_320.0),
    ]

    for model_name, tp, seq_len, batch, expected_weight, expected_kv in cases:
        model = MODELS[model_name]
        weight = stage1_model.b_weight_footprint(model, tp)
        kv = stage1_model.b_kv_footprint(model, tp, seq_len, batch)

        assert weight == expected_weight
        assert kv == expected_kv
        assert weight + kv == b_local_per_token(model, tp, seq_len, batch)


def test_batched_decode_reuses_weights_and_scales_kv_remote_and_compute():
    batch_one = build_decode_signature("Llama-7B", tp=2, seq_len=128, batch=1)
    batch_two = build_decode_signature("Llama-7B", tp=2, seq_len=128, batch=2)

    # Per GPU: 7e9 weight bytes; KV is 33,554,432 bytes per batch element.
    assert batch_one.b_local_bytes == 7_033_554_432.0
    assert batch_two.b_local_bytes == 7_067_108_864.0
    assert batch_two.b_remote_bytes == 2.0 * batch_one.b_remote_bytes
    assert batch_two.compute_flops == 2.0 * batch_one.compute_flops


def test_llama2_70b_gqa_reduces_only_kv_term_to_one_eighth_of_mha():
    model = MODELS["Llama-70B"]

    b_local = b_local_per_token(model, tp=4, seq_len=4096, batch=1)
    b_weight = 35_000_000_000.0
    old_mha_kv = 2_684_354_560.0
    expected_gqa_kv = 335_544_320.0

    assert b_local - b_weight == expected_gqa_kv
    assert old_mha_kv / (b_local - b_weight) == 8.0


def test_llama2_7b_mha_kv_traffic_is_unchanged():
    b_local = b_local_per_token(MODELS["Llama-7B"], tp=2, seq_len=128, batch=1)

    assert b_local == 7_033_554_432.0


def test_rejects_tp_that_does_not_evenly_partition_kv_heads():
    with pytest.raises(ValueError, match="n_kv_heads .* evenly divisible by tp"):
        b_local_per_token(MODELS["Llama-70B"], tp=3, seq_len=4096, batch=1)


def test_r_remote_only_splits_b_local_traffic():
    signature = build_decode_signature("Llama-70B", tp=4, seq_len=4096)
    low_remote = split_traffic(signature, r_remote=0.1, r=0.25)
    high_remote = split_traffic(signature, r_remote=0.5, r=0.25)

    assert low_remote.local_hbm_bytes == pytest.approx(0.9 * signature.b_local_bytes)
    assert low_remote.side_hbm_bytes == pytest.approx(0.1 * signature.b_local_bytes)
    assert high_remote.local_hbm_bytes == pytest.approx(0.5 * signature.b_local_bytes)
    assert high_remote.side_hbm_bytes == pytest.approx(0.5 * signature.b_local_bytes)
    assert high_remote.h2h_bytes == low_remote.h2h_bytes
    assert high_remote.gpu_direct_bytes == low_remote.gpu_direct_bytes
    # Weight traffic exists in both attention and MLP stages, so every
    # semantic stage has a side request when r_remote is nonzero.
    assert low_remote.n_side_requests == 2 * MODELS["Llama-70B"]["n_layers"]


def test_r_only_splits_collective_traffic():
    signature = build_decode_signature("Llama-70B", tp=4, seq_len=4096)
    h2h_only = split_traffic(signature, r_remote=0.25, r=0.0)
    hybrid = split_traffic(signature, r_remote=0.25, r=0.75)
    direct_only = split_traffic(signature, r_remote=0.25, r=1.0)

    assert hybrid.h2h_bytes == pytest.approx(0.25 * signature.b_remote_bytes)
    assert hybrid.gpu_direct_bytes == pytest.approx(0.75 * signature.b_remote_bytes)
    assert h2h_only.gpu_direct_bytes == 0.0
    assert direct_only.h2h_bytes == 0.0
    assert h2h_only.local_hbm_bytes == hybrid.local_hbm_bytes
    assert h2h_only.side_hbm_bytes == hybrid.side_hbm_bytes


def test_zero_side_fraction_disables_side_request_count():
    signature = build_decode_signature("Llama-7B", tp=2, seq_len=128)

    streams = split_traffic(signature, r_remote=0.0, r=0.0)

    assert streams.n_side_requests == 0


@pytest.mark.parametrize(("r_remote", "r"), [(-0.1, 0.0), (0.0, 1.1)])
def test_rejects_invalid_traffic_split_fractions(r_remote, r):
    signature = build_decode_signature("Llama-7B", tp=2, seq_len=128)

    with pytest.raises(ValueError):
        split_traffic(signature, r_remote=r_remote, r=r)
