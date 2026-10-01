import importlib
from pathlib import Path

import pytest


CONFIG_TEXT = """
mode: layer_prefetch
architectures:
  - name: B0-2.5D
    kind: b0_2_5d
    lane_count: 8
    local_memory:
      capacity_bytes: 48e9
      bw_bps: 1.2e12
      latency_s: 1.0e-7
      energy_pj_per_bit: 4.33
  - name: B2-uHBM
    kind: b2_uhbm
    lane_count: 8
    local_memory:
      capacity_bytes: 12e9
      bw_bps: 2.4e12
      latency_s: 0.0
      energy_pj_per_bit: 4.33
    side_hbm:
      capacity_bytes: 48e9
      bw_bps: 1.2e12
      latency_s: 5.0e-8
      energy_pj_per_bit: 4.33
    side_link:
      bw_bps: 256e9
      latency_s: 2.0e-8
      energy_pj_per_bit: 0.54
capacity_memory:
  device:
    capacity_bytes: 64e9
    read_bw_bps: 256e9
    read_latency_s: 1.5e-7
    read_energy_pj_per_bit: 6.0
  link:
    bw_bps: 128e9
    latency_s: 4.0e-8
    energy_pj_per_bit: 1.0
    distance_mm: 20.0
execution:
  gpu_peak_flops: 990e12
  r: 0.0
  comm_mode: materialized
  h2h_bw_bps: 336e9
  h2h_latency_s: 2.0e-8
  gpu_direct_bw_bps: 768e9
  gpu_direct_latency_s: 2.0e-8
  base_die_latency_s: 5.0e-9
  include_materialized_endpoints: true
prefetch:
  weight_residency_policy: [persistent_if_fit, streaming]
  weight_chunk_bytes: [1048576]
  kv_chunk_bytes: [262144]
  kv_writeback_chunk_bytes: [65536]
  max_inflight_chunks_per_lane: [1, 4, null]
workload:
  run_mode: [prefill_only, decode_only, prefill_decode]
  start_state: [cold_start, steady_state]
  model: Llama-70B
  tp: 4
  pp: 1
  seq_len: 4096
  batch: [1, 8]
  decode_tokens: [1, 8, 32]
"""


def _module():
    return importlib.import_module("uhbm_simulator.prefetch_config")


def _write(tmp_path: Path, text: str = CONFIG_TEXT) -> Path:
    path = tmp_path / "prefetch.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_eight_lane_b0_b2_and_technology_neutral_profiles(tmp_path):
    """Wrong capacities or conflated device/link parameters must fail."""
    config = _module().load_prefetch_config(_write(tmp_path))

    assert config.mode == "layer_prefetch"
    by_kind = {item.kind: item for item in config.architectures}
    assert by_kind["b0_2_5d"].lane_count == 8
    assert by_kind["b0_2_5d"].local_memory.capacity_bytes == 48e9
    assert by_kind["b0_2_5d"].side_hbm is None
    assert by_kind["b2_uhbm"].lane_count == 8
    assert by_kind["b2_uhbm"].local_memory.capacity_bytes == 12e9
    assert by_kind["b2_uhbm"].side_hbm.capacity_bytes == 48e9
    assert config.capacity_memory.device.read_bw_bps == 256e9
    assert config.capacity_memory.link.bw_bps == 128e9
    assert config.capacity_memory.link.distance_mm == 20.0
    assert config.execution.gpu_peak_flops == 990e12
    assert config.execution.h2h_bw_bps == 336e9
    assert config.execution.comm_mode == "materialized"


def test_loads_independent_chunk_depth_and_workload_sweep_axes(tmp_path):
    """Collapsing the three chunk sizes or dropping null depth must fail."""
    config = _module().load_prefetch_config(_write(tmp_path))

    assert config.prefetch.weight_residency_policy_values == (
        "persistent_if_fit",
        "streaming",
    )
    assert config.prefetch.weight_chunk_bytes_values == (1048576,)
    assert config.prefetch.kv_chunk_bytes_values == (262144,)
    assert config.prefetch.kv_writeback_chunk_bytes_values == (65536,)
    assert config.prefetch.max_inflight_chunks_per_lane_values == (1, 4, None)
    assert config.workload.run_mode_values == (
        "prefill_only",
        "decode_only",
        "prefill_decode",
    )
    assert config.workload.start_state_values == ("cold_start", "steady_state")
    assert config.workload.batch_values == (1, 8)
    assert config.workload.decode_tokens_values == (1, 8, 32)


@pytest.mark.parametrize("lane_count", [1, 4, 7, 9])
def test_rejects_non_eight_lane_architectures(tmp_path, lane_count):
    text = CONFIG_TEXT.replace("lane_count: 8", f"lane_count: {lane_count}", 1)
    with pytest.raises(_module().PrefetchConfigError, match="lane_count.*8"):
        _module().load_prefetch_config(_write(tmp_path, text))


def test_rejects_b2_without_paired_side_hbm_and_link(tmp_path):
    start = CONFIG_TEXT.index("    side_hbm:\n")
    end = CONFIG_TEXT.index("capacity_memory:\n")
    text = CONFIG_TEXT[:start] + CONFIG_TEXT[end:]

    with pytest.raises(_module().PrefetchConfigError, match="b2_uhbm.*side_hbm"):
        _module().load_prefetch_config(_write(tmp_path, text))


@pytest.mark.parametrize(
    "field",
    ["weight_chunk_bytes", "kv_chunk_bytes", "kv_writeback_chunk_bytes"],
)
def test_rejects_nonpositive_chunk_sizes(tmp_path, field):
    text = CONFIG_TEXT.replace(f"{field}: [", f"{field}: [0, ")
    with pytest.raises(_module().PrefetchConfigError, match=field):
        _module().load_prefetch_config(_write(tmp_path, text))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("prefill_only, decode_only, prefill_decode", "training", "run_mode"),
        ("cold_start, steady_state", "lukewarm", "start_state"),
        ("persistent_if_fit, streaming", "partial_lru", "weight_residency"),
    ],
)
def test_rejects_unknown_modes(tmp_path, old, new, message):
    text = CONFIG_TEXT.replace(old, new)
    with pytest.raises(_module().PrefetchConfigError, match=message):
        _module().load_prefetch_config(_write(tmp_path, text))


def test_rejects_capacity_memory_too_small_for_one_lane_weight_backing(tmp_path):
    text = CONFIG_TEXT.replace("capacity_bytes: 64e9", "capacity_bytes: 1e9")
    with pytest.raises(
        _module().PrefetchConfigError,
        match="capacity_memory.*weight backing",
    ):
        _module().load_prefetch_config(_write(tmp_path, text))
