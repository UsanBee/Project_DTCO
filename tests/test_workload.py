import pytest

from stage1_model import MODELS, b_local_per_token, b_remote_per_token
from uhbm_simulator.workload import build_decode_signature, split_traffic


def test_decode_signature_reuses_stage1_traffic_functions():
    signature = build_decode_signature("Llama-70B", tp=4, seq_len=4096, batch=1)
    expected_local = b_local_per_token(MODELS["Llama-70B"], 4, 4096, 1)
    expected_remote, expected_messages = b_remote_per_token(
        MODELS["Llama-70B"], 4, 1
    )

    assert signature.b_local_bytes == expected_local
    assert signature.b_remote_bytes == expected_remote
    assert signature.n_messages == expected_messages
    assert signature.n_side_requests == MODELS["Llama-70B"]["n_layers"]


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
    assert low_remote.n_side_requests == signature.n_side_requests


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
