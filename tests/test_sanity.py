from uhbm_simulator.sanity import run_sanity_checks


def test_all_spec_sanity_checks_pass():
    report = run_sanity_checks()

    assert report.checks == {
        "stage1_break_even_matches": True,
        "higher_uhbm_bw_lowers_tpot": True,
        "higher_h2h_bw_lowers_comm_time": True,
        "decode_is_more_latency_sensitive": True,
        "higher_r_reduces_h2h_dependency": True,
        "r_one_matches_conventional_b0": True,
        "higher_n_pillar_lowers_h2h_bw": True,
        "thermal_scenarios_order_compute_time": True,
    }
