import math

import pytest

from stage1_model import MODELS, required_bw_d2d
from uhbm_simulator.analysis import empirical_break_even_bw, h2h_feasibility


@pytest.mark.parametrize(
    ("model_name", "tp", "seq_len", "k_b"),
    [
        ("Llama-7B", 2, 4096, 1.5),
        ("Llama-70B", 4, 4096, 2.0),
        ("Llama-70B", 8, 16384, 3.0),
    ],
)
def test_empirical_stage2_break_even_matches_stage1_closed_form(
    model_name, tp, seq_len, k_b
):
    expected = required_bw_d2d(
        MODELS[model_name], tp=tp, seq_len=seq_len, k_B=k_b
    )
    empirical = empirical_break_even_bw(
        model_name=model_name,
        tp=tp,
        seq_len=seq_len,
        k_b=k_b,
        batch=1,
    )

    assert math.isfinite(expected)
    assert empirical == pytest.approx(expected, rel=1e-6)


def test_r_zero_feasibility_checks_bandwidth_and_latency_constraints():
    inadequate = h2h_feasibility(
        model_name="Llama-70B",
        tp=4,
        seq_len=4096,
        k_b=2.0,
        batch=1,
        actual_bw_bps=1.0,
        actual_latency_s=1.0,
    )

    assert not inadequate.bandwidth_feasible
    assert not inadequate.latency_feasible
    assert not inadequate.architecture_feasible


def test_empirical_search_has_no_one_megabyte_per_second_floor():
    expected = required_bw_d2d(
        MODELS["Llama-7B"], tp=2, seq_len=100_000_000, k_B=1e9
    )

    empirical = empirical_break_even_bw(
        model_name="Llama-7B",
        tp=2,
        seq_len=100_000_000,
        k_b=1e9,
    )

    assert expected < 1e6
    assert empirical == pytest.approx(expected, rel=1e-6)
