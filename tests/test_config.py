from pathlib import Path

import pytest

import uhbm_simulator.config as config_module
from uhbm_simulator.config import (
    DEFAULT_GPU_DIRECT_BW_BPS,
    DEFAULT_GPU_DIRECT_LATENCY_S,
    DEFAULT_H2H_LANE_COUNT,
    DEFAULT_N_PILLAR_MAX,
    ConfigError,
    h2h_bandwidth_bps,
    load_config,
)


CONFIG_TEXT = """
baseline: B2-hybrid
comm_mode: materialized
thermal_scenario: no_throttling
r: [0, 0.25, 0.5, 0.75, 1.0]
channels:
  local_hbm:
    stack:
      bw_bps: 1.2e12
      capacity_bytes: 16e9
    count: [4]
    max_count: 4
    address_mapping:
      policy: uniform
      weights: null
  side_hbm:
    bw_bps: 4.8e12
    latency_s: 50e-9
  h2h:
    lane_rate_gbps: [13, 32, 48]
    latency_s: 20e-9
    n_pillar: [4, 8, 16, 32]
    n_pillar_max: 64
  gpu_direct:
    bw_bps: TBD
    latency_s: TBD
workload:
  model: Llama-70B
  tp: 4
  pp: 1
  seq_len: 4096
  batch: 1
  r_remote: [0, 0.1, 0.25, 0.5]
"""


def write_config(tmp_path: Path, text: str = CONFIG_TEXT) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_spec_yaml_and_resolves_documented_placeholders(tmp_path):
    config = load_config(write_config(tmp_path))

    assert config.baseline == "B2-hybrid"
    assert config.side_hbm_path_values == ("shared_h2h",)
    assert config.comm_mode == "materialized"
    assert config.r_values == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert config.workload.phase == "decode"
    assert config.channels.local_hbm.stack.bw_bps == 1.2e12
    assert config.channels.local_hbm.stack.capacity_bytes == 16e9
    assert config.channels.local_hbm.count_values == (4,)
    assert config.channels.local_hbm.max_count == 4
    assert config.channels.local_hbm.address_mapping.policy == "uniform"
    assert config.channels.local_hbm.address_mapping.weights is None
    assert config.channels.h2h.n_pillar == (4, 8, 16, 32)
    assert config.channels.gpu_direct.bw_bps == DEFAULT_GPU_DIRECT_BW_BPS
    assert config.channels.gpu_direct.latency_s == DEFAULT_GPU_DIRECT_LATENCY_S


def test_side_hbm_path_accepts_a_research_sweep_and_rejects_unknown_values(tmp_path):
    swept = CONFIG_TEXT.replace(
        "baseline: B2-hybrid",
        "baseline: B2-hybrid\nside_hbm_path: [shared_h2h, independent]",
    )
    assert load_config(write_config(tmp_path, swept)).side_hbm_path_values == (
        "shared_h2h",
        "independent",
    )

    invalid = CONFIG_TEXT.replace(
        "baseline: B2-hybrid",
        "baseline: B2-hybrid\nside_hbm_path: average_hops",
    )
    with pytest.raises(ConfigError, match="side_hbm_path"):
        load_config(write_config(tmp_path, invalid))


def test_energy_defaults_and_ucie_rate_mapping_are_loaded(tmp_path):
    config = load_config(write_config(tmp_path))

    assert config.channels.local_hbm.stack.energy_pj_per_bit == 4.33
    assert config.channels.local_hbm.stack.hbm_idle_power_w == 0.4
    assert config.channels.side_hbm.energy_pj_per_bit == 4.33
    assert config.channels.side_hbm.link_energy_pj_per_bit == 10.0
    assert config.channels.gpu_direct.energy_pj_per_bit == 0.54
    assert config_module.h2h_energy_pj_per_bit(12.0) == 0.25
    assert config_module.h2h_energy_pj_per_bit(16.0) == 0.30
    assert config_module.h2h_energy_pj_per_bit(24.0) == 0.30
    assert config_module.h2h_energy_pj_per_bit(32.0) == 0.30


def test_explicit_hbm_idle_and_side_link_energy_values_are_loaded(tmp_path):
    explicit = CONFIG_TEXT.replace(
        "capacity_bytes: 16e9",
        "capacity_bytes: 16e9\n      hbm_idle_power_w: 2.0",
    ).replace(
        "latency_s: 50e-9",
        "latency_s: 50e-9\n    link_energy_pj_per_bit: 20.0",
    )

    config = load_config(write_config(tmp_path, explicit))

    assert config.channels.local_hbm.stack.hbm_idle_power_w == 2.0
    assert config.channels.side_hbm.link_energy_pj_per_bit == 20.0


def test_scalar_batch_is_normalized_to_singleton_sweep(tmp_path):
    config = load_config(write_config(tmp_path))

    assert config.workload.batch_values == (1,)


def test_batch_list_is_preserved_as_sweep_values(tmp_path):
    config = load_config(
        write_config(tmp_path, CONFIG_TEXT.replace("batch: 1", "batch: [1, 4, 8]"))
    )

    assert config.workload.batch_values == (1, 4, 8)


@pytest.mark.parametrize("batch", ["[]", "0", "[1, 0]", "[1, 1.5]", "[true]"])
def test_rejects_empty_nonpositive_or_noninteger_batch_sweep(tmp_path, batch):
    invalid = CONFIG_TEXT.replace("batch: 1", f"batch: {batch}")

    with pytest.raises(ConfigError, match="batch"):
        load_config(write_config(tmp_path, invalid))


@pytest.mark.parametrize(
    "invalid_config",
    [
        CONFIG_TEXT.replace(
            "capacity_bytes: 16e9", "capacity_bytes: 16e9\n      energy_pj_per_bit: -1"
        ),
        CONFIG_TEXT.replace(
            "latency_s: 50e-9", "latency_s: 50e-9\n    energy_pj_per_bit: .nan"
        ),
        CONFIG_TEXT.replace(
            "latency_s: TBD", "latency_s: TBD\n    energy_pj_per_bit: 0"
        ),
    ],
)
def test_rejects_nonpositive_or_nonfinite_channel_energy(tmp_path, invalid_config):
    with pytest.raises(ConfigError, match="energy"):
        load_config(write_config(tmp_path, invalid_config))


@pytest.mark.parametrize("invalid_value", ["0", "-0.1", ".nan", ".inf", "true"])
@pytest.mark.parametrize(
    ("anchor", "field"),
    [
        ("capacity_bytes: 16e9", "hbm_idle_power_w"),
        ("latency_s: 50e-9", "link_energy_pj_per_bit"),
    ],
)
def test_rejects_invalid_hbm_background_or_side_link_energy(
    tmp_path, anchor, field, invalid_value
):
    invalid = CONFIG_TEXT.replace(
        anchor,
        f"{anchor}\n{'      ' if field == 'hbm_idle_power_w' else '    '}"
        f"{field}: {invalid_value}",
    )

    with pytest.raises(ConfigError, match=field):
        load_config(write_config(tmp_path, invalid))


def test_h2h_bandwidth_strictly_decreases_as_n_pillar_increases():
    bandwidths = [
        h2h_bandwidth_bps(
            lane_rate_gbps=48,
            n_pillar=n_pillar,
            n_pillar_max=64,
            lane_count=DEFAULT_H2H_LANE_COUNT,
        )
        for n_pillar in (4, 8, 16, 32)
    ]

    assert all(a > b for a, b in zip(bandwidths, bandwidths[1:]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("comm_mode", "unknown"),
        ("thermal_scenario", "thermal_solver"),
    ],
)
def test_rejects_modes_outside_scope_lock(tmp_path, field, value):
    invalid = CONFIG_TEXT.replace(f"{field}: materialized", f"{field}: {value}")
    if field == "thermal_scenario":
        invalid = CONFIG_TEXT.replace(
            "thermal_scenario: no_throttling", f"thermal_scenario: {value}"
        )

    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, invalid))


def test_rejects_out_of_range_traffic_fractions(tmp_path):
    invalid = CONFIG_TEXT.replace("r: [0, 0.25, 0.5, 0.75, 1.0]", "r: [1.1]")

    with pytest.raises(ConfigError, match="r"):
        load_config(write_config(tmp_path, invalid))


def test_accepts_optional_prefill_phase(tmp_path):
    prefill = CONFIG_TEXT.replace("workload:\n", "workload:\n  phase: prefill\n")

    config = load_config(write_config(tmp_path, prefill))

    assert config.workload.phase == "prefill"


def test_rejects_unknown_workload_phase(tmp_path):
    invalid = CONFIG_TEXT.replace("workload:\n", "workload:\n  phase: training\n")

    with pytest.raises(ConfigError, match="phase"):
        load_config(write_config(tmp_path, invalid))


def test_resolves_n_pillar_max_tbd_placeholder(tmp_path):
    placeholder = CONFIG_TEXT.replace("n_pillar_max: 64", "n_pillar_max: TBD")

    config = load_config(write_config(tmp_path, placeholder))

    assert config.channels.h2h.n_pillar_max == DEFAULT_N_PILLAR_MAX


def test_rejects_unmodeled_pipeline_parallelism(tmp_path):
    unsupported = CONFIG_TEXT.replace("pp: 1", "pp: 2")

    with pytest.raises(ConfigError, match="pp=1"):
        load_config(write_config(tmp_path, unsupported))


@pytest.mark.parametrize("count", [0, 5])
def test_rejects_local_hbm_count_outside_physical_limit(tmp_path, count):
    invalid = CONFIG_TEXT.replace("count: [4]", f"count: [{count}]")

    with pytest.raises(ConfigError, match="count"):
        load_config(write_config(tmp_path, invalid))


def test_rejects_unknown_local_hbm_mapping_policy(tmp_path):
    invalid = CONFIG_TEXT.replace("policy: uniform", "policy: hashed")

    with pytest.raises(ConfigError, match="policy"):
        load_config(write_config(tmp_path, invalid))


@pytest.mark.parametrize(
    "weights",
    [
        "[1, 0, 0]",
        "[1, -1, 1, 1]",
        "[0, 0, 0, 0]",
    ],
)
def test_rejects_invalid_weighted_local_hbm_mapping(tmp_path, weights):
    invalid = CONFIG_TEXT.replace("policy: uniform", "policy: weighted").replace(
        "weights: null", f"weights: {weights}"
    )

    with pytest.raises(ConfigError, match="weights"):
        load_config(write_config(tmp_path, invalid))


def test_rejects_weighted_mapping_across_multiple_stack_counts(tmp_path):
    invalid = (
        CONFIG_TEXT.replace("count: [4]", "count: [1, 4]")
        .replace("policy: uniform", "policy: weighted")
        .replace("weights: null", "weights: [1, 1, 1, 1]")
    )

    with pytest.raises(ConfigError, match="single count"):
        load_config(write_config(tmp_path, invalid))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("max_count: 4", "max_count: 5"),
        ("count: [4]", "count: [1.9]"),
        ("count: [4]", "count: [true]"),
        ("max_count: 4", "max_count: 4.9"),
    ],
)
def test_rejects_nonphysical_or_noninteger_stack_counts(tmp_path, old, new):
    invalid = CONFIG_TEXT.replace(old, new)

    with pytest.raises(ConfigError, match="count"):
        load_config(write_config(tmp_path, invalid))


@pytest.mark.parametrize("nonfinite", [".nan", ".inf", "-.inf"])
def test_rejects_nonfinite_weighted_mapping_values(tmp_path, nonfinite):
    invalid = (
        CONFIG_TEXT.replace("policy: uniform", "policy: weighted")
        .replace("weights: null", f"weights: [{nonfinite}, 1, 1, 1]")
    )

    with pytest.raises(ConfigError, match="weights"):
        load_config(write_config(tmp_path, invalid))


@pytest.mark.parametrize("field", ["bw_bps", "capacity_bytes"])
@pytest.mark.parametrize("nonfinite", [".nan", ".inf"])
def test_rejects_nonfinite_per_stack_parameters(tmp_path, field, nonfinite):
    invalid = CONFIG_TEXT.replace(
        f"{field}: {'1.2e12' if field == 'bw_bps' else '16e9'}",
        f"{field}: {nonfinite}",
        1,
    )

    with pytest.raises(ConfigError, match="bandwidth|capacity"):
        load_config(write_config(tmp_path, invalid))
