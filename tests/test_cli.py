from uhbm_simulator.cli import main

from test_sweep import SMALL_CONFIG


def test_cli_runs_config_sweep_and_writes_csv(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    output = tmp_path / "results.csv"
    config.write_text(SMALL_CONFIG, encoding="utf-8")

    exit_code = main(["--config", str(config), "--output", str(output)])

    assert exit_code == 0
    assert output.exists()
    assert "4 design points" in capsys.readouterr().out


def test_cli_sanity_prints_every_check(capsys):
    exit_code = main(["--sanity"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "stage1_break_even_matches: PASS" in output
    assert "higher_n_pillar_lowers_h2h_bw: PASS" in output
