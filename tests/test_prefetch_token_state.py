import math

import pytest

from stage1_model import MODELS, b_kv_footprint
from uhbm_simulator.resources import TrafficClass

from test_layer_prefetch_simulator import _config


def test_fresh_prefill_generates_kv_without_historical_kv_prefetch():
    """Reading nonexistent prior-session KV during fresh prefill must fail."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    result = run_prefetch_simulation(
        _config(run_mode="prefill_only"), detailed_trace=True
    )

    assert result.path_bytes.get(TrafficClass.SIDE_HBM_READ, 0.0) == 0.0
    assert result.path_bytes[TrafficClass.UHBM_TO_SIDE] > 0.0
    token = result.trace.token_timings[0]
    assert token.phase == "prefill"
    assert token.kv_resident_bytes_per_lane_start == 0.0
    assert token.kv_resident_bytes_per_lane_end > 0.0


def test_decode_only_starts_with_configured_historical_kv_resident():
    """Decode-only must not start from an empty KV backing tier."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    config = _config(run_mode="decode_only")
    result = run_prefetch_simulation(config, detailed_trace=True)
    token = result.trace.token_timings[0]
    expected = b_kv_footprint(MODELS["Llama-7B"], 2, 64, 1) / 8.0

    assert token.kv_resident_bytes_per_lane_start == expected
    assert token.kv_resident_bytes_per_lane_end > expected


def test_prefill_decode_transfers_generated_kv_state_into_decode():
    """Resetting KV state between phases must fail this boundary check."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    result = run_prefetch_simulation(
        _config(run_mode="prefill_decode", decode_tokens=2),
        detailed_trace=True,
    )
    prefill, first_decode = result.trace.token_timings[:2]

    assert prefill.phase == "prefill"
    assert first_decode.phase == "decode"
    assert first_decode.kv_resident_bytes_per_lane_start == (
        prefill.kv_resident_bytes_per_lane_end
    )


def test_new_kv_writeback_finishes_before_next_token_layer_prefetch():
    """A next-token read racing an incomplete side-HBM writeback must fail."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    result = run_prefetch_simulation(
        _config(run_mode="decode_only", decode_tokens=2),
        detailed_trace=True,
    )
    layers_per_token = MODELS["Llama-7B"]["n_layers"]
    first_token = result.trace.layer_timings[:layers_per_token]
    second_token = result.trace.layer_timings[layers_per_token:]

    for previous, current in zip(first_token, second_token):
        assert previous.writeback_end_s is not None
        assert current.prefetch_launch_s >= previous.writeback_end_s


def test_decode_kv_residency_grows_one_increment_per_token():
    """Losing or double-counting token KV growth must fail conservation."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    result = run_prefetch_simulation(
        _config(run_mode="decode_only", decode_tokens=3),
        detailed_trace=True,
    )
    tokens = result.trace.token_timings
    increments = [
        token.kv_resident_bytes_per_lane_end
        - token.kv_resident_bytes_per_lane_start
        for token in tokens
    ]

    assert len(tokens) == 3
    assert increments[0] > 0.0
    assert increments == pytest.approx([increments[0]] * 3)
    assert tokens[1].kv_resident_bytes_per_lane_start == pytest.approx(
        tokens[0].kv_resident_bytes_per_lane_end
    )


def test_cold_start_measures_layer_zero_preparation():
    """Fabricating a pre-ready layer-zero buffer would hide cold TTFT."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    result = run_prefetch_simulation(
        _config(start_state="cold_start"), detailed_trace=True
    )
    first = result.trace.layer_timings[0]
    token = result.trace.token_timings[0]

    assert first.prefetch_launch_s == token.started_s
    assert first.compute_start_s > token.started_s
    assert result.first_token_latency_s == token.latency_s


def test_steady_state_executes_but_excludes_one_warmup_token():
    """Counting warmup or fabricating state without running it must fail."""
    from uhbm_simulator.prefetch_simulator import run_prefetch_simulation

    result = run_prefetch_simulation(
        _config(start_state="steady_state", decode_tokens=2),
        detailed_trace=True,
    )
    tokens = result.trace.token_timings

    assert len(tokens) == 3
    assert tokens[0].measured is False
    assert all(token.measured for token in tokens[1:])
    assert result.token_latencies_s == pytest.approx(
        tuple(token.latency_s for token in tokens[1:])
    )
    assert tokens[1].started_s >= tokens[0].ended_s
    assert all(math.isfinite(value) and value > 0.0 for value in result.token_latencies_s)

