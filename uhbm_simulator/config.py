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

# Chatterjee et al., "Architecting an Energy-Efficient DRAM System For GPUs,"
# HPCA 2017: incremental HBM datapath energy at 50% toggle =
# 1.48 + 2.31 + 0.54 pJ/bit. This is transfer-dependent circuit energy, not
# an effective P_total/BW value containing background/static power.
DEFAULT_HBM_ENERGY_PJ_PER_BIT = 4.33
# Arunkumar et al., "MCM-GPU," ISCA 2017: 28 nm, 20 Gb/s GRS signaling.
DEFAULT_GPU_DIRECT_ENERGY_PJ_PER_BIT = 0.54
# Arunkumar et al., "Understanding the Future of Energy Efficiency in
# Multi-Module GPUs," HPCA 2019. These separate literature anchors happen to
# converge to the same 0.54 pJ/bit value as the MCM-GPU direct-link constant
# above, but they describe a different field and therefore remain distinct.
ARUNKUMAR2019_ON_PACKAGE_LINK_ENERGY_PJ_PER_BIT = 0.54
ARUNKUMAR2019_ON_BOARD_LINK_ENERGY_PJ_PER_BIT = 10.0

# Primary nominal anchor: Stow et al., "Power Profiling of Modern Die-Stacked
# Memory," IEEE CAL 2019, Fig. 3(a), 4-Hi HBM2 at 1000 MHz. The paper defines
# Idle as clock-enabled with no access; approximately 0.4 W/stack is a visual
# chart reading (+/-0.1 W uncertainty), not a scalar tabulated in the text.
# Idle is the closest state because this simulator keeps every configured HBM
# stack powered and ready for the full run and does not model self-refresh or
# power-down transitions. The value is nominal idle/background power, not pure
# transistor leakage and not a measured uHBM constant.
STOW2019_HBM2_4HI_IDLE_STATE_POWER_W_APPROX = 0.4
STOW2019_HBM2_4HI_MAX_TOTAL_POWER_W = 15.6
# Larimi et al., DATE 2021, Section III-A-2: idle HBM consumes nearly one-third
# of full-load power on a different measured platform.
LARIMI2021_IDLE_TO_FULL_LOAD_RATIO = 1.0 / 3.0
# Cross-methodology high reference / stress point only. This is not a physical
# upper bound: it combines numbers from different platforms and methodologies.
# At about 5.2 W it is roughly 13x Stow's direct Idle-state chart reading. The
# available sources do not isolate whether platform, measurement boundary,
# workload/state definition, PHY inclusion, or methodology dominates the gap.
CROSS_METHODOLOGY_HIGH_REFERENCE_IDLE_POWER_W = (
    STOW2019_HBM2_4HI_MAX_TOTAL_POWER_W
    * LARIMI2021_IDLE_TO_FULL_LOAD_RATIO
)
DEFAULT_HBM_IDLE_POWER_W_PER_STACK = (
    STOW2019_HBM2_4HI_IDLE_STATE_POWER_W_APPROX
)
# Stack-height scaling is deliberately absent. Stow reports about 1.4x scaling
# of static components from 4-Hi to 8-Hi and a larger total Idle bar, but this
# round neither extracts nor depends on a scalar for that specific 8-Hi bar.
# UCIe 1.0 Advanced Package specification at 0.5 V, as re-cited by Chiu et
# al., Nature Electronics 2024: <16 GT/s and >=16 GT/s operating regions.
UCIe_ENERGY_PJ_PER_BIT_BY_REGION = {"below_16_gtps": 0.25, "at_least_16_gtps": 0.30}

BASELINES = {"B0", "B1", "B2-pure", "B2-hybrid"}
COMM_MODES = {"materialized", "cut_through"}
SIDE_HBM_PATHS = {"shared_h2h", "independent", "pillar_bypass"}
SIDE_HBM_OWNER_SCOPES = {"all_remote", "adjacent"}
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
    energy_pj_per_bit: float
    hbm_idle_power_w: float


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
    energy_pj_per_bit: float
    link_energy_pj_per_bit: float


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
    energy_pj_per_bit: float


@dataclass(frozen=True)
class PillarLinkOperatingPointConfig:
    """One paired BW/latency/energy sensitivity point for the D-3 path."""

    name: str
    bw_bps: float
    latency_s: float
    energy_pj_per_bit: float


@dataclass(frozen=True)
class PillarLinkConfig:
    operating_points: tuple[PillarLinkOperatingPointConfig, ...]


@dataclass(frozen=True)
class ChannelsConfig:
    local_hbm: LocalHBMConfig
    side_hbm: SideHBMConfig
    h2h: H2HConfig
    gpu_direct: GPUDirectConfig
    pillar_link: PillarLinkConfig | None = None


@dataclass(frozen=True)
class WorkloadConfig:
    phase: str
    model: str
    tp: int
    pp: int
    seq_len: int
    batch_values: tuple[int, ...]
    # Placement fractions for b_local in capacity-unconstrained side HBM.
    # The model assumes the access fraction follows the resident placement.
    r_remote_values: tuple[float, ...]


@dataclass(frozen=True)
class SimulationConfig:
    baseline: str
    side_hbm_path_values: tuple[str, ...]
    comm_mode: str
    thermal_scenario: str
    r_values: tuple[float, ...]
    channels: ChannelsConfig
    workload: WorkloadConfig
    side_hbm_owner_scope_values: tuple[str, ...] = ("all_remote",)
    physical_adjacency_by_tp: dict[int, tuple[tuple[int, ...], ...]] | None = None


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _fractions(value: Any, name: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in _as_tuple(value))
    if not values or any(item < 0.0 or item > 1.0 for item in values):
        raise ConfigError(f"{name} values must be within [0, 1]")
    return values


def _side_hbm_paths(value: Any) -> tuple[str, ...]:
    values = tuple(str(item) for item in _as_tuple(value))
    if not values or any(item not in SIDE_HBM_PATHS for item in values):
        raise ConfigError(
            f"side_hbm_path values must be one of {sorted(SIDE_HBM_PATHS)}"
        )
    return values


def _side_hbm_owner_scopes(value: Any) -> tuple[str, ...]:
    values = tuple(str(item) for item in _as_tuple(value))
    if not values or any(item not in SIDE_HBM_OWNER_SCOPES for item in values):
        raise ConfigError(
            "side_hbm_owner_scope values must be one of "
            f"{sorted(SIDE_HBM_OWNER_SCOPES)}"
        )
    return values


def _validate_adjacency_for_tp(
    adjacency: tuple[tuple[int, ...], ...], tp: int
) -> None:
    if len(adjacency) != tp:
        raise ConfigError(
            "physical_adjacency must have exactly one entry per TP tile"
        )
    for requester, owners in enumerate(adjacency):
        if not owners:
            raise ConfigError("each tile must have at least one adjacent owner")
        if len(set(owners)) != len(owners):
            raise ConfigError(
                "physical_adjacency neighbor entries must be unique"
            )
        for owner in owners:
            if owner < 0 or owner >= tp:
                raise ConfigError("physical_adjacency tile index is out of range")
            if owner == requester:
                raise ConfigError("physical_adjacency cannot contain a self edge")
    for requester, owners in enumerate(adjacency):
        for owner in owners:
            if requester not in adjacency[owner]:
                raise ConfigError("physical_adjacency must be symmetric")


def _load_pillar_link(raw: Any) -> PillarLinkConfig:
    if not isinstance(raw, dict):
        raise ConfigError("channels.pillar_link must be a mapping")
    points_raw = _required(raw, "operating_points")
    if not isinstance(points_raw, list) or not points_raw:
        raise ConfigError("pillar_link.operating_points must be a nonempty list")
    points = []
    names = set()
    for point_raw in points_raw:
        if not isinstance(point_raw, dict):
            raise ConfigError("each pillar operating point must be a mapping")
        name = str(_required(point_raw, "name"))
        if not name or name in names:
            raise ConfigError("pillar operating-point names must be nonempty and unique")
        names.add(name)
        latency = float(_required(point_raw, "latency_s"))
        if not math.isfinite(latency) or latency < 0.0:
            raise ConfigError("pillar_link latency_s must be finite and nonnegative")
        points.append(
            PillarLinkOperatingPointConfig(
                name=name,
                bw_bps=_positive_float(
                    _required(point_raw, "bw_bps"), "pillar_link.bw_bps"
                ),
                latency_s=latency,
                energy_pj_per_bit=_positive_float(
                    _required(point_raw, "energy_pj_per_bit"),
                    "pillar_link.energy_pj_per_bit",
                ),
            )
        )

    return PillarLinkConfig(tuple(points))


def _load_physical_adjacency(
    adjacency_raw: Any,
) -> dict[int, tuple[tuple[int, ...], ...]]:
    if not isinstance(adjacency_raw, dict):
        raise ConfigError("physical_adjacency must be a TP mapping")
    adjacency_by_tp = {}
    for raw_tp, tile_map in adjacency_raw.items():
        tp = _strict_int(int(raw_tp), "physical_adjacency TP")
        if tp <= 1 or not isinstance(tile_map, dict):
            raise ConfigError(
                "physical_adjacency TP keys must exceed one and map tiles"
            )
        expected = set(range(tp))
        try:
            actual = {int(key) for key in tile_map}
        except (TypeError, ValueError) as error:
            raise ConfigError("physical_adjacency tile keys must be integers") from error
        if actual != expected:
            raise ConfigError(
                "physical_adjacency must define exactly every tile 0..TP-1"
            )
        adjacency = []
        for requester in range(tp):
            raw_neighbors = tile_map.get(requester, tile_map.get(str(requester)))
            if not isinstance(raw_neighbors, list):
                raise ConfigError("physical_adjacency neighbors must be lists")
            neighbors = tuple(
                _strict_int(owner, "physical_adjacency owner")
                for owner in raw_neighbors
            )
            adjacency.append(neighbors)
        normalized = tuple(adjacency)
        _validate_adjacency_for_tp(normalized, tp)
        adjacency_by_tp[tp] = normalized
    return adjacency_by_tp


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


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{name} must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigError(f"{name} must be positive and finite")
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


def h2h_energy_pj_per_bit(lane_rate_gbps: float) -> float:
    """Return the UCIe 1.0 Advanced Package energy for a lane-rate region."""
    if not math.isfinite(lane_rate_gbps) or lane_rate_gbps <= 0.0:
        raise ConfigError("H2H lane rate must be positive and finite")
    region = "below_16_gtps" if lane_rate_gbps < 16.0 else "at_least_16_gtps"
    return UCIe_ENERGY_PJ_PER_BIT_BY_REGION[region]


def _required(mapping: dict[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise ConfigError(f"missing required config key: {key}") from error


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return value


def _positive_int_values(value: Any, name: str) -> tuple[int, ...]:
    values = tuple(_strict_int(item, name) for item in _as_tuple(value))
    if not values or any(item <= 0 for item in values):
        raise ConfigError(f"{name} values must be positive integers")
    return values


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
            energy_pj_per_bit=_positive_float(
                stack_raw.get(
                    "energy_pj_per_bit", DEFAULT_HBM_ENERGY_PJ_PER_BIT
                ),
                "local_hbm.stack.energy_pj_per_bit",
            ),
            # All configured stacks stay powered and ready for the full run.
            # HBM self-refresh/power-down transitions and stack-height scaling
            # are intentionally outside the Stage 2 model.
            hbm_idle_power_w=_positive_float(
                stack_raw.get(
                    "hbm_idle_power_w", DEFAULT_HBM_IDLE_POWER_W_PER_STACK
                ),
                "local_hbm.stack.hbm_idle_power_w",
            ),
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
    side_hbm_path_values = _side_hbm_paths(
        raw.get("side_hbm_path", "shared_h2h")
    )
    side_hbm_owner_scope_values = _side_hbm_owner_scopes(
        raw.get("side_hbm_owner_scope", "all_remote")
    )
    pillar_link = (
        _load_pillar_link(channels_raw.get("pillar_link"))
        if channels_raw.get("pillar_link") is not None
        else None
    )
    physical_adjacency_by_tp = (
        _load_physical_adjacency(raw.get("physical_adjacency"))
        if raw.get("physical_adjacency") is not None
        else None
    )

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
    workload_tp = int(_required(workload_raw, "tp"))
    if "pillar_bypass" in side_hbm_path_values:
        if side_hbm_owner_scope_values != ("adjacent",):
            raise ConfigError(
                "pillar_bypass requires side_hbm_owner_scope: adjacent"
            )
        if pillar_link is None:
            raise ConfigError(
                "channels.pillar_link is required when pillar_bypass is selected"
            )
    if "adjacent" in side_hbm_owner_scope_values and workload_tp > 1:
        if physical_adjacency_by_tp is None:
            raise ConfigError(
                "physical_adjacency is required for adjacent ownership"
            )
        if workload_tp not in physical_adjacency_by_tp:
            raise ConfigError(
                f"physical_adjacency is not defined for TP={workload_tp}"
            )
    return SimulationConfig(
        baseline=baseline,
        side_hbm_path_values=side_hbm_path_values,
        comm_mode=comm_mode,
        thermal_scenario=thermal_scenario,
        r_values=_fractions(_required(raw, "r"), "r"),
        channels=ChannelsConfig(
            local_hbm=_load_local_hbm(local_raw),
            side_hbm=SideHBMConfig(
                bw_bps=float(_required(side_raw, "bw_bps")),
                latency_s=float(_required(side_raw, "latency_s")),
                energy_pj_per_bit=_positive_float(
                    side_raw.get(
                        "energy_pj_per_bit", DEFAULT_HBM_ENERGY_PJ_PER_BIT
                    ),
                    "side_hbm.energy_pj_per_bit",
                ),
                # Nominal on-board/PCB NUMA-crossing reference from Arunkumar
                # et al., HPCA 2019. uHBM's actual integration domain is not
                # confirmed: 0.54 pJ/bit is the on-package anchor. Treat this
                # scalar as a nominal reference and use sensitivity.py's
                # range, not 10 pJ/bit as ground truth.
                # TODO(태현): replace after Stage 3/4 physical calibration.
                link_energy_pj_per_bit=_positive_float(
                    side_raw.get(
                        "link_energy_pj_per_bit",
                        ARUNKUMAR2019_ON_BOARD_LINK_ENERGY_PJ_PER_BIT,
                    ),
                    "side_hbm.link_energy_pj_per_bit",
                ),
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
                energy_pj_per_bit=_positive_float(
                    direct_raw.get(
                        "energy_pj_per_bit",
                        DEFAULT_GPU_DIRECT_ENERGY_PJ_PER_BIT,
                    ),
                    "gpu_direct.energy_pj_per_bit",
                ),
            ),
            pillar_link=pillar_link,
        ),
        workload=WorkloadConfig(
            phase=phase,
            model=str(_required(workload_raw, "model")),
            tp=workload_tp,
            pp=pp,
            seq_len=int(_required(workload_raw, "seq_len")),
            batch_values=_positive_int_values(
                _required(workload_raw, "batch"), "workload.batch"
            ),
            r_remote_values=_fractions(
                _required(workload_raw, "r_remote"), "r_remote"
            ),
        ),
        side_hbm_owner_scope_values=side_hbm_owner_scope_values,
        physical_adjacency_by_tp=physical_adjacency_by_tp,
    )
