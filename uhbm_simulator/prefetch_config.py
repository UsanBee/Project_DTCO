"""Configuration schema for the dedicated layer-prefetch experiments.

This loader is intentionally separate from :mod:`uhbm_simulator.config`.
The existing Stage 2 configs retain their legacy semantics, while this schema
describes the symmetric eight-lane B0/B2 memory hierarchy used by the
layer-ready prefetch engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml

from stage1_model import MODELS, b_weight_footprint


LANE_COUNT = 8
ARCHITECTURE_KINDS = {"b0_2_5d", "b2_uhbm"}
WEIGHT_RESIDENCY_POLICIES = {"persistent_if_fit", "streaming"}
RUN_MODES = {"prefill_only", "decode_only", "prefill_decode"}
START_STATES = {"cold_start", "steady_state"}


class PrefetchConfigError(ValueError):
    """Raised when a layer-prefetch YAML file violates its schema."""


@dataclass(frozen=True)
class MemoryDeviceConfig:
    capacity_bytes: float
    bw_bps: float
    latency_s: float
    energy_pj_per_bit: float


@dataclass(frozen=True)
class LinkConfig:
    bw_bps: float
    latency_s: float
    energy_pj_per_bit: float
    distance_mm: float | None = None


@dataclass(frozen=True)
class CapacityMemoryDeviceConfig:
    capacity_bytes: float
    read_bw_bps: float
    read_latency_s: float
    read_energy_pj_per_bit: float


@dataclass(frozen=True)
class CapacityMemoryConfig:
    device: CapacityMemoryDeviceConfig
    link: LinkConfig


@dataclass(frozen=True)
class ArchitectureConfig:
    name: str
    kind: str
    lane_count: int
    local_memory: MemoryDeviceConfig
    side_hbm: MemoryDeviceConfig | None = None
    side_link: LinkConfig | None = None


@dataclass(frozen=True)
class PrefetchPolicyConfig:
    weight_residency_policy_values: tuple[str, ...]
    weight_chunk_bytes_values: tuple[int, ...]
    kv_chunk_bytes_values: tuple[int, ...]
    kv_writeback_chunk_bytes_values: tuple[int, ...]
    max_inflight_chunks_per_lane_values: tuple[int | None, ...]


@dataclass(frozen=True)
class PrefetchWorkloadConfig:
    run_mode_values: tuple[str, ...]
    start_state_values: tuple[str, ...]
    model: str
    tp: int
    pp: int
    seq_len: int
    batch_values: tuple[int, ...]
    decode_tokens_values: tuple[int, ...]


@dataclass(frozen=True)
class PrefetchSimulationConfig:
    mode: str
    architectures: tuple[ArchitectureConfig, ...]
    capacity_memory: CapacityMemoryConfig
    prefetch: PrefetchPolicyConfig
    workload: PrefetchWorkloadConfig


def _required(mapping: dict[str, Any], key: str, path: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise PrefetchConfigError(f"missing required config key: {path}.{key}") from error


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PrefetchConfigError(f"{path} must be a mapping")
    return value


def _values(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, list) else (value,)


def _positive_float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise PrefetchConfigError(f"{path} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PrefetchConfigError(f"{path} must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise PrefetchConfigError(f"{path} must be positive and finite")
    return result


def _nonnegative_float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise PrefetchConfigError(f"{path} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PrefetchConfigError(f"{path} must be numeric") from error
    if not math.isfinite(result) or result < 0.0:
        raise PrefetchConfigError(f"{path} must be nonnegative and finite")
    return result


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PrefetchConfigError(f"{path} must be a positive integer")
    return value


def _positive_int_values(value: Any, path: str) -> tuple[int, ...]:
    values = tuple(_positive_int(item, path) for item in _values(value))
    if not values:
        raise PrefetchConfigError(f"{path} must not be empty")
    return values


def _enum_values(value: Any, allowed: set[str], path: str) -> tuple[str, ...]:
    values = tuple(str(item) for item in _values(value))
    if not values or any(item not in allowed for item in values):
        raise PrefetchConfigError(
            f"{path} values must be one of {sorted(allowed)}"
        )
    return values


def _load_memory_device(raw: Any, path: str) -> MemoryDeviceConfig:
    data = _mapping(raw, path)
    return MemoryDeviceConfig(
        capacity_bytes=_positive_float(
            _required(data, "capacity_bytes", path), f"{path}.capacity_bytes"
        ),
        bw_bps=_positive_float(_required(data, "bw_bps", path), f"{path}.bw_bps"),
        latency_s=_nonnegative_float(
            _required(data, "latency_s", path), f"{path}.latency_s"
        ),
        energy_pj_per_bit=_positive_float(
            _required(data, "energy_pj_per_bit", path),
            f"{path}.energy_pj_per_bit",
        ),
    )


def _load_link(raw: Any, path: str, *, allow_distance: bool) -> LinkConfig:
    data = _mapping(raw, path)
    distance = None
    if allow_distance:
        distance = _positive_float(
            _required(data, "distance_mm", path), f"{path}.distance_mm"
        )
    return LinkConfig(
        bw_bps=_positive_float(_required(data, "bw_bps", path), f"{path}.bw_bps"),
        latency_s=_nonnegative_float(
            _required(data, "latency_s", path), f"{path}.latency_s"
        ),
        energy_pj_per_bit=_positive_float(
            _required(data, "energy_pj_per_bit", path),
            f"{path}.energy_pj_per_bit",
        ),
        distance_mm=distance,
    )


def _load_architecture(raw: Any, index: int) -> ArchitectureConfig:
    path = f"architectures[{index}]"
    data = _mapping(raw, path)
    kind = str(_required(data, "kind", path))
    if kind not in ARCHITECTURE_KINDS:
        raise PrefetchConfigError(
            f"{path}.kind must be one of {sorted(ARCHITECTURE_KINDS)}"
        )
    lane_count = _positive_int(_required(data, "lane_count", path), f"{path}.lane_count")
    if lane_count != LANE_COUNT:
        raise PrefetchConfigError(
            f"{path}.lane_count must be exactly {LANE_COUNT} for the symmetric model"
        )

    side_hbm = None
    side_link = None
    if kind == "b2_uhbm":
        if "side_hbm" not in data or "side_link" not in data:
            raise PrefetchConfigError(
                "b2_uhbm requires paired side_hbm and side_link fields"
            )
        side_hbm = _load_memory_device(data["side_hbm"], f"{path}.side_hbm")
        side_link = _load_link(data["side_link"], f"{path}.side_link", allow_distance=False)
    elif "side_hbm" in data or "side_link" in data:
        raise PrefetchConfigError("b0_2_5d must not define side_hbm or side_link")

    return ArchitectureConfig(
        name=str(_required(data, "name", path)),
        kind=kind,
        lane_count=lane_count,
        local_memory=_load_memory_device(data.get("local_memory"), f"{path}.local_memory"),
        side_hbm=side_hbm,
        side_link=side_link,
    )


def load_prefetch_config(path: str | Path) -> PrefetchSimulationConfig:
    """Load the standalone eight-lane layer-prefetch experiment schema."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    root = _mapping(raw, "configuration")
    mode = str(_required(root, "mode", "configuration"))
    if mode != "layer_prefetch":
        raise PrefetchConfigError("mode must be layer_prefetch")

    architectures_raw = _required(root, "architectures", "configuration")
    if not isinstance(architectures_raw, list) or not architectures_raw:
        raise PrefetchConfigError("architectures must be a nonempty list")
    architectures = tuple(
        _load_architecture(item, index)
        for index, item in enumerate(architectures_raw)
    )
    kinds = tuple(item.kind for item in architectures)
    if len(set(kinds)) != len(kinds):
        raise PrefetchConfigError("architecture kinds must be unique")

    capacity_raw = _mapping(
        _required(root, "capacity_memory", "configuration"), "capacity_memory"
    )
    device_raw = _mapping(
        _required(capacity_raw, "device", "capacity_memory"),
        "capacity_memory.device",
    )
    capacity_memory = CapacityMemoryConfig(
        device=CapacityMemoryDeviceConfig(
            capacity_bytes=_positive_float(
                _required(device_raw, "capacity_bytes", "capacity_memory.device"),
                "capacity_memory.device.capacity_bytes",
            ),
            read_bw_bps=_positive_float(
                _required(device_raw, "read_bw_bps", "capacity_memory.device"),
                "capacity_memory.device.read_bw_bps",
            ),
            read_latency_s=_nonnegative_float(
                _required(device_raw, "read_latency_s", "capacity_memory.device"),
                "capacity_memory.device.read_latency_s",
            ),
            read_energy_pj_per_bit=_positive_float(
                _required(
                    device_raw,
                    "read_energy_pj_per_bit",
                    "capacity_memory.device",
                ),
                "capacity_memory.device.read_energy_pj_per_bit",
            ),
        ),
        link=_load_link(
            _required(capacity_raw, "link", "capacity_memory"),
            "capacity_memory.link",
            allow_distance=True,
        ),
    )

    prefetch_raw = _mapping(
        _required(root, "prefetch", "configuration"), "prefetch"
    )
    depths: list[int | None] = []
    for value in _values(
        _required(prefetch_raw, "max_inflight_chunks_per_lane", "prefetch")
    ):
        depths.append(
            None
            if value is None
            else _positive_int(value, "prefetch.max_inflight_chunks_per_lane")
        )
    if not depths:
        raise PrefetchConfigError(
            "prefetch.max_inflight_chunks_per_lane must not be empty"
        )
    prefetch = PrefetchPolicyConfig(
        weight_residency_policy_values=_enum_values(
            _required(prefetch_raw, "weight_residency_policy", "prefetch"),
            WEIGHT_RESIDENCY_POLICIES,
            "prefetch.weight_residency_policy",
        ),
        weight_chunk_bytes_values=_positive_int_values(
            _required(prefetch_raw, "weight_chunk_bytes", "prefetch"),
            "prefetch.weight_chunk_bytes",
        ),
        kv_chunk_bytes_values=_positive_int_values(
            _required(prefetch_raw, "kv_chunk_bytes", "prefetch"),
            "prefetch.kv_chunk_bytes",
        ),
        kv_writeback_chunk_bytes_values=_positive_int_values(
            _required(prefetch_raw, "kv_writeback_chunk_bytes", "prefetch"),
            "prefetch.kv_writeback_chunk_bytes",
        ),
        max_inflight_chunks_per_lane_values=tuple(depths),
    )

    workload_raw = _mapping(
        _required(root, "workload", "configuration"), "workload"
    )
    model = str(_required(workload_raw, "model", "workload"))
    if model not in MODELS:
        raise PrefetchConfigError(f"workload.model must be one of {sorted(MODELS)}")
    workload = PrefetchWorkloadConfig(
        run_mode_values=_enum_values(
            _required(workload_raw, "run_mode", "workload"),
            RUN_MODES,
            "workload.run_mode",
        ),
        start_state_values=_enum_values(
            _required(workload_raw, "start_state", "workload"),
            START_STATES,
            "workload.start_state",
        ),
        model=model,
        tp=_positive_int(_required(workload_raw, "tp", "workload"), "workload.tp"),
        pp=_positive_int(_required(workload_raw, "pp", "workload"), "workload.pp"),
        seq_len=_positive_int(
            _required(workload_raw, "seq_len", "workload"), "workload.seq_len"
        ),
        batch_values=_positive_int_values(
            _required(workload_raw, "batch", "workload"), "workload.batch"
        ),
        decode_tokens_values=_positive_int_values(
            _required(workload_raw, "decode_tokens", "workload"),
            "workload.decode_tokens",
        ),
    )
    if workload.pp != 1:
        raise PrefetchConfigError("workload.pp must be 1 in the Stage 2 prefetch model")

    # The capacity-memory device is per lane. It must hold that lane's full
    # striped backing copy of the TP-sharded model weights.
    weight_backing_per_lane = (
        b_weight_footprint(MODELS[workload.model], workload.tp) / LANE_COUNT
    )
    if capacity_memory.device.capacity_bytes < weight_backing_per_lane:
        raise PrefetchConfigError(
            "capacity_memory device is too small for one-lane weight backing"
        )

    return PrefetchSimulationConfig(
        mode=mode,
        architectures=architectures,
        capacity_memory=capacity_memory,
        prefetch=prefetch,
        workload=workload,
    )
