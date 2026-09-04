"""YAML configuration loading and Stage 2 placeholder derivations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


# MCM-GPU(Arunkumar et al., ISCA 2017)의 on-package inter-GPU-module 직접 링크
# "low-effort, low-cost, low-energy" 설계점: 768 GB/s. 기존 900GB/s는 stage1_model.py의
# BW_INTERTILE_0을 그대로 재사용한 미확인 placeholder였음 -- 이번에 문헌값으로 교체.
DEFAULT_GPU_DIRECT_BW_BPS = 768e9
# 위 논문의 32-cycle hop latency와 자릿수는 맞지만(≈1.6GHz 가정 시 20ns), 정확한 클럭을
# 확인하지 못해 기존 20ns를 그대로 유지. TODO(태현): 정확한 클럭 가정 확인 후 재계산.
DEFAULT_GPU_DIRECT_LATENCY_S = 20e-9
# 128 = UCIe Advanced Package 모듈 폭(x64 lane/module) x 2모듈로 해석 가능 -- 근거는
# 있으나 실제 설계에서 몇 모듈을 쓸지는 미정. TODO(태현): Stage 3에서 실제 모듈 수 확정.
DEFAULT_H2H_LANE_COUNT = 128
# TODO(태현): 근거 없음 -- 공개 문헌으로 이 프로젝트의 base die shoreline/via 면적
# 예산을 특정할 수 없었음. Stage 3 floorplan이 나오면 교체할 것.
DEFAULT_N_PILLAR_MAX = 64

BASELINES = {"B0", "B1", "B2-pure", "B2-hybrid"}
COMM_MODES = {"materialized", "cut_through"}
THERMAL_SCENARIOS = {
    "no_throttling",
    "mild_throttling",
    "severe_throttling",
}
MAX_LOCAL_HBM_STACKS = 4


class ConfigError(ValueError):
    """Raised when a Stage 2 YAML configuration violates the spec."""


@dataclass(frozen=True)
class LocalHBMStackConfig:
    bw_bps: float
    capacity_bytes: float


@dataclass(frozen=True)
class AddressMappingConfig:
    policy: str
    weights: tuple[float, ...] | None


@dataclass(frozen=True)
class LocalHBMConfig:
    stack: LocalHBMStackConfig
    count_values: tuple[int, ...]
    max_count: int
    address_mapping: AddressMappingConfig


@dataclass(frozen=True)
class SideHBMConfig:
    bw_bps: float
    latency_s: float


@dataclass(frozen=True)
class H2HConfig:
    lane_rate_gbps: tuple[float, ...]
    latency_s: float
    n_pillar: tuple[int, ...]
    n_pillar_max: int


@dataclass(frozen=True)
class GPUDirectConfig:
    bw_bps: float
    latency_s: float


@dataclass(frozen=True)
class ChannelsConfig:
    local_hbm: LocalHBMConfig
    side_hbm: SideHBMConfig
    h2h: H2HConfig
    gpu_direct: GPUDirectConfig


@dataclass(frozen=True)
class WorkloadConfig:
    phase: str
    model: str
    tp: int
    pp: int
    seq_len: int
    batch: int
    r_remote_values: tuple[float, ...]


@dataclass(frozen=True)
class SimulationConfig:
    baseline: str
    comm_mode: str
    thermal_scenario: str
    r_values: tuple[float, ...]
    channels: ChannelsConfig
    workload: WorkloadConfig


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _fractions(value: Any, name: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in _as_tuple(value))
    if not values or any(item < 0.0 or item > 1.0 for item in values):
        raise ConfigError(f"{name} values must be within [0, 1]")
    return values


def _placeholder_number(value: Any, default: float, name: str) -> float:
    if isinstance(value, str) and value.strip().upper() == "TBD":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{name} must be numeric or TBD") from error
    if result <= 0:
        raise ConfigError(f"{name} must be positive")
    return result


def h2h_bandwidth_bps(
    lane_rate_gbps: float,
    n_pillar: int,
    n_pillar_max: int,
    lane_count: int = DEFAULT_H2H_LANE_COUNT,
) -> float:
    """Return the Stage 2 linear PDN/H2H resource-competition placeholder."""
    if lane_rate_gbps <= 0 or lane_count <= 0:
        raise ConfigError("H2H lane rate and lane count must be positive")
    if n_pillar < 0 or n_pillar >= n_pillar_max:
        raise ConfigError("n_pillar must satisfy 0 <= n_pillar < n_pillar_max")

    # TODO(태현): Stage 3 HFSS/ADS + PDN 결과로 이 선형 근사를 교체한다.
    available_fraction = 1.0 - (n_pillar / n_pillar_max)
    raw_bw_bps = lane_rate_gbps * 1e9 * lane_count / 8.0
    return raw_bw_bps * available_fraction


def _required(mapping: dict[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise ConfigError(f"missing required config key: {key}") from error


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return value


def _load_local_hbm(local_raw: dict[str, Any]) -> LocalHBMConfig:
    stack_raw = _required(local_raw, "stack")
    count_values = tuple(
        _strict_int(item, "local_hbm.count")
        for item in _as_tuple(_required(local_raw, "count"))
    )
    max_count = _strict_int(_required(local_raw, "max_count"), "local_hbm.max_count")
    if max_count < 1 or max_count > MAX_LOCAL_HBM_STACKS:
        raise ConfigError(
            f"local_hbm.max_count must satisfy 1 <= max_count <= "
            f"{MAX_LOCAL_HBM_STACKS}"
        )
    if not count_values or any(
        count < 1 or count > max_count for count in count_values
    ):
        raise ConfigError(
            "local_hbm.count values must satisfy 1 <= count <= max_count"
        )

    mapping_raw = _required(local_raw, "address_mapping")
    policy = str(_required(mapping_raw, "policy"))
    if policy not in {"uniform", "weighted"}:
        raise ConfigError(
            "local_hbm.address_mapping.policy must be uniform or weighted"
        )

    weights: tuple[float, ...] | None = None
    if policy == "weighted":
        if len(count_values) != 1:
            raise ConfigError(
                "weighted local_hbm mapping requires a single count value"
            )
        raw_weights = _required(mapping_raw, "weights")
        if raw_weights is None or not isinstance(raw_weights, list):
            raise ConfigError("weighted local_hbm mapping requires weights")
        weights = tuple(float(weight) for weight in raw_weights)
        if (
            len(weights) != count_values[0]
            or any(
                not math.isfinite(weight) or weight < 0.0
                for weight in weights
            )
            or max(weights, default=0.0) <= 0.0
        ):
            raise ConfigError(
                "local_hbm.address_mapping.weights must match count, be "
                "nonnegative, and contain at least one positive value"
            )

    stack_bw_bps = float(_required(stack_raw, "bw_bps"))
    stack_capacity_bytes = float(_required(stack_raw, "capacity_bytes"))
    if (
        not math.isfinite(stack_bw_bps)
        or not math.isfinite(stack_capacity_bytes)
        or stack_bw_bps <= 0.0
        or stack_capacity_bytes <= 0.0
    ):
        raise ConfigError(
            "local_hbm stack bandwidth and capacity must be positive and finite"
        )
    return LocalHBMConfig(
        stack=LocalHBMStackConfig(
            bw_bps=stack_bw_bps,
            capacity_bytes=stack_capacity_bytes,
        ),
        count_values=count_values,
        max_count=max_count,
        address_mapping=AddressMappingConfig(policy=policy, weights=weights),
    )


def load_config(path: str | Path) -> SimulationConfig:
    """Load and validate the YAML format defined by spec chapter 9."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")

    baseline = str(_required(raw, "baseline"))
    comm_mode = str(_required(raw, "comm_mode"))
    thermal_scenario = str(_required(raw, "thermal_scenario"))
    if baseline not in BASELINES:
        raise ConfigError(f"baseline must be one of {sorted(BASELINES)}")
    if comm_mode not in COMM_MODES:
        raise ConfigError(f"comm_mode must be one of {sorted(COMM_MODES)}")
    if thermal_scenario not in THERMAL_SCENARIOS:
        raise ConfigError(
            f"thermal_scenario must be one of {sorted(THERMAL_SCENARIOS)}"
        )

    channels_raw = _required(raw, "channels")
    local_raw = _required(channels_raw, "local_hbm")
    side_raw = _required(channels_raw, "side_hbm")
    h2h_raw = _required(channels_raw, "h2h")
    direct_raw = _required(channels_raw, "gpu_direct")

    h2h = H2HConfig(
        lane_rate_gbps=tuple(
            float(item) for item in _as_tuple(_required(h2h_raw, "lane_rate_gbps"))
        ),
        latency_s=float(_required(h2h_raw, "latency_s")),
        n_pillar=tuple(
            int(item) for item in _as_tuple(_required(h2h_raw, "n_pillar"))
        ),
        n_pillar_max=int(
            _placeholder_number(
                _required(h2h_raw, "n_pillar_max"),
                DEFAULT_N_PILLAR_MAX,
                "h2h.n_pillar_max",
            )
        ),
    )
    for lane_rate in h2h.lane_rate_gbps:
        for n_pillar in h2h.n_pillar:
            h2h_bandwidth_bps(lane_rate, n_pillar, h2h.n_pillar_max)

    workload_raw = _required(raw, "workload")
    phase = str(workload_raw.get("phase", "decode"))
    if phase not in {"prefill", "decode"}:
        raise ConfigError("workload.phase must be prefill or decode")
    pp = int(_required(workload_raw, "pp"))
    # TODO(태현): pipeline-parallel scheduling model이 정해지면 pp>1을 지원한다.
    if pp != 1:
        raise ConfigError("Stage 2 currently supports pp=1 only")
    return SimulationConfig(
        baseline=baseline,
        comm_mode=comm_mode,
        thermal_scenario=thermal_scenario,
        r_values=_fractions(_required(raw, "r"), "r"),
        channels=ChannelsConfig(
            local_hbm=_load_local_hbm(local_raw),
            side_hbm=SideHBMConfig(
                bw_bps=float(_required(side_raw, "bw_bps")),
                latency_s=float(_required(side_raw, "latency_s")),
            ),
            h2h=h2h,
            gpu_direct=GPUDirectConfig(
                # TODO(태현): 실제 GPU-direct BW/latency 측정값으로 교체한다.
                bw_bps=_placeholder_number(
                    _required(direct_raw, "bw_bps"),
                    DEFAULT_GPU_DIRECT_BW_BPS,
                    "gpu_direct.bw_bps",
                ),
                latency_s=_placeholder_number(
                    _required(direct_raw, "latency_s"),
                    DEFAULT_GPU_DIRECT_LATENCY_S,
                    "gpu_direct.latency_s",
                ),
            ),
        ),
        workload=WorkloadConfig(
            phase=phase,
            model=str(_required(workload_raw, "model")),
            tp=int(_required(workload_raw, "tp")),
            pp=pp,
            seq_len=int(_required(workload_raw, "seq_len")),
            batch=int(_required(workload_raw, "batch")),
            r_remote_values=_fractions(
                _required(workload_raw, "r_remote"), "r_remote"
            ),
        ),
    )
