"""Workload-layer traffic signatures derived from the Stage 1 model."""

from __future__ import annotations

from dataclasses import dataclass
import math

from stage1_model import (
    MODELS,
    activation_footprint_bytes,
    activation_traffic_bytes,
    attention_parameter_fraction,
    attention_flops_prefill,
    b_kv_footprint,
    b_remote_per_token,
    b_weight_footprint,
)


class SideHBMCapacityError(ValueError):
    """Raised when capacity-driven placement exceeds side-HBM capacity."""

    def __init__(
        self,
        required_capacity_bytes: float,
        capacity_bytes: float,
    ) -> None:
        super().__init__(
            "side_hbm resident footprint exceeds configured capacity: "
            f"required={required_capacity_bytes}, capacity={capacity_bytes}"
        )
        self.required_capacity_bytes = required_capacity_bytes
        self.capacity_bytes = capacity_bytes


@dataclass(frozen=True)
class StageSignature:
    """Placement-independent traffic and work for one semantic sub-layer."""

    layer_index: int
    sublayer: str
    weight_traffic_bytes: float
    kv_traffic_bytes: float
    activation_traffic_bytes: float
    compute_flops: float
    collective_bytes: float


@dataclass(frozen=True)
class ClassPlacementFractions:
    """Local resident fractions retained for class-specific traffic splitting."""

    weight_local: float
    kv_local: float
    activation_local: float


@dataclass(frozen=True)
class StageTrafficStreams:
    """Placement-aware resource traffic for one attention or MLP stage."""

    layer_index: int
    sublayer: str
    weight_local_bytes: float
    weight_side_bytes: float
    kv_local_bytes: float
    kv_side_bytes: float
    activation_local_bytes: float
    activation_side_bytes: float
    compute_flops: float
    h2h_bytes: float
    gpu_direct_bytes: float

    @property
    def local_hbm_bytes(self) -> float:
        return (
            self.weight_local_bytes
            + self.kv_local_bytes
            + self.activation_local_bytes
        )

    @property
    def side_hbm_bytes(self) -> float:
        return (
            self.weight_side_bytes
            + self.kv_side_bytes
            + self.activation_side_bytes
        )


@dataclass(frozen=True)
class TrafficSignature:
    phase: str
    b_local_bytes: float
    b_remote_bytes: float
    n_messages: int
    compute_flops: float
    token_count: int = 1
    # Legacy/custom aggregate-signature scheduling hint only. LLM signatures
    # carry semantic_stages, for which side request count is derived after
    # placement from the stages with nonzero side traffic.
    n_side_requests: int = 1
    # Resident local-memory state used only by capacity accounting. None keeps
    # legacy/custom signatures compatible by falling back to b_local_bytes.
    local_footprint_bytes: float | None = None
    weight_footprint_bytes: float | None = None
    kv_footprint_bytes: float | None = None
    activation_footprint_bytes: float | None = None
    weight_traffic_bytes: float | None = None
    kv_traffic_bytes: float | None = None
    activation_traffic_bytes_value: float | None = None
    semantic_stages: tuple[StageSignature, ...] | None = None


@dataclass(frozen=True)
class TrafficStreams:
    local_hbm_bytes: float
    side_hbm_bytes: float
    h2h_bytes: float
    gpu_direct_bytes: float
    n_messages: int
    n_side_requests: int
    effective_r_remote: float
    local_resident_footprint_bytes: float
    side_resident_footprint_bytes: float
    class_placement: ClassPlacementFractions | None
    semantic_stages: tuple[StageTrafficStreams, ...] | None


def _split_evenly(total: float, count: int) -> tuple[float, ...]:
    """Split a total with a final residual so math.fsum conserves the input."""
    if count <= 0:
        raise ValueError("semantic stage split count must be positive")
    if count == 1:
        return (total,)
    share = total / count
    parts = [share] * (count - 1)
    parts.append(total - math.fsum(parts))
    return tuple(parts)


def _build_semantic_stages(
    model,
    *,
    weight_traffic: float,
    kv_traffic: float,
    activation_traffic: float,
    linear_compute_flops: float,
    attention_compute_flops: float,
    collective_bytes: float,
) -> tuple[StageSignature, ...]:
    """Build alternating attention/MLP stages before memory placement."""
    n_layers = model["n_layers"]
    attention_fraction = attention_parameter_fraction(model)

    attention_weight_total = weight_traffic * attention_fraction
    mlp_weight_total = weight_traffic - attention_weight_total
    attention_linear_total = linear_compute_flops * attention_fraction
    mlp_linear_total = linear_compute_flops - attention_linear_total

    attention_weights = _split_evenly(attention_weight_total, n_layers)
    mlp_weights = _split_evenly(mlp_weight_total, n_layers)
    kv_by_layer = _split_evenly(kv_traffic, n_layers)
    activation_by_stage = _split_evenly(activation_traffic, 2 * n_layers)
    attention_linear = _split_evenly(attention_linear_total, n_layers)
    mlp_linear = _split_evenly(mlp_linear_total, n_layers)
    attention_extra = _split_evenly(attention_compute_flops, n_layers)

    # In this project's current Megatron-style aggregate abstraction, the
    # attention and MLP logical collectives use the same hidden-state payload,
    # so B_remote/(2L) is an exact conservation split within this abstraction.
    # Sequence parallelism, fusion, or implementation-specific communication
    # can make the two payloads differ on real systems; that is not modeled.
    collective_by_stage = _split_evenly(collective_bytes, 2 * n_layers)

    stages = []
    for layer in range(n_layers):
        attention_index = 2 * layer
        stages.append(
            StageSignature(
                layer_index=layer,
                sublayer="attention",
                weight_traffic_bytes=attention_weights[layer],
                kv_traffic_bytes=kv_by_layer[layer],
                activation_traffic_bytes=activation_by_stage[attention_index],
                compute_flops=(
                    attention_linear[layer] + attention_extra[layer]
                ),
                collective_bytes=collective_by_stage[attention_index],
            )
        )
        mlp_index = attention_index + 1
        stages.append(
            StageSignature(
                layer_index=layer,
                sublayer="mlp",
                weight_traffic_bytes=mlp_weights[layer],
                kv_traffic_bytes=0.0,
                activation_traffic_bytes=activation_by_stage[mlp_index],
                compute_flops=mlp_linear[layer],
                collective_bytes=collective_by_stage[mlp_index],
            )
        )
    return tuple(stages)


def build_decode_signature(
    model_name: str,
    tp: int,
    seq_len: int,
    batch: int = 1,
) -> TrafficSignature:
    """Build one decode-token signature using Stage 1 traffic functions."""
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name}")
    if tp < 1 or seq_len < 0 or batch < 1:
        raise ValueError("tp and batch must be positive; seq_len cannot be negative")

    model = MODELS[model_name]
    weight_footprint = b_weight_footprint(model, tp)
    kv_footprint = b_kv_footprint(model, tp, seq_len, batch)
    activation_footprint = 0.0
    weight_traffic = weight_footprint
    kv_traffic = kv_footprint
    activation_traffic = 0.0
    b_local = weight_traffic + kv_traffic + activation_traffic
    b_remote, n_messages = b_remote_per_token(model, tp, batch)
    # TODO(태현): 실제 kernel efficiency를 반영한 FLOP model로 교체한다.
    # Decode attention FLOPs remain unchanged in this round for scope control.
    # This is a known approximation, not zero physical attention computation:
    # per-token attention over the KV cache grows as O(seq_len) at long context
    # and is not modeled here.
    compute_flops = 2.0 * model["n_params"] * batch / tp
    semantic_stages = _build_semantic_stages(
        model,
        weight_traffic=weight_traffic,
        kv_traffic=kv_traffic,
        activation_traffic=activation_traffic,
        linear_compute_flops=compute_flops,
        attention_compute_flops=0.0,
        collective_bytes=b_remote,
    )
    return TrafficSignature(
        phase="decode",
        b_local_bytes=b_local,
        b_remote_bytes=b_remote,
        n_messages=n_messages,
        compute_flops=compute_flops,
        local_footprint_bytes=(
            weight_footprint + kv_footprint + activation_footprint
        ),
        weight_footprint_bytes=weight_footprint,
        kv_footprint_bytes=kv_footprint,
        activation_footprint_bytes=activation_footprint,
        weight_traffic_bytes=weight_traffic,
        kv_traffic_bytes=kv_traffic,
        activation_traffic_bytes_value=activation_traffic,
        semantic_stages=semantic_stages,
    )


def build_prefill_signature(
    model_name: str,
    tp: int,
    seq_len: int,
    batch: int = 1,
    c_causal: float = 1.0,
    c_act_traffic: float = 0.0,
    c_act_footprint: float = 0.0,
) -> TrafficSignature:
    """Build a compute-dominant prefill signature for a full sequence."""
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name}")
    if tp < 1 or seq_len < 1 or batch < 1:
        raise ValueError("tp, seq_len, and batch must be positive")

    model = MODELS[model_name]
    # TODO(태현): Stage 3/4 data로 prefill memory traffic approximation을 보정한다.
    weight_footprint = b_weight_footprint(model, tp)
    kv_footprint = b_kv_footprint(model, tp, seq_len, batch)
    activation_footprint = activation_footprint_bytes(
        model,
        tp,
        seq_len,
        batch,
        c_act_footprint=c_act_footprint,
    )
    weight_traffic = weight_footprint
    kv_traffic = kv_footprint
    activation_traffic = activation_traffic_bytes(
        model,
        tp,
        seq_len,
        batch,
        c_act_traffic=c_act_traffic,
    )
    local_footprint = weight_footprint + kv_footprint + activation_footprint
    b_local = weight_traffic + kv_traffic + activation_traffic
    b_remote_per_token_value, n_messages = b_remote_per_token(model, tp, batch)
    linear_flops = 2.0 * model["n_params"] * seq_len * batch / tp
    attention_flops = attention_flops_prefill(
        model,
        tp,
        seq_len,
        batch,
        c_causal=c_causal,
    )
    semantic_stages = _build_semantic_stages(
        model,
        weight_traffic=weight_traffic,
        kv_traffic=kv_traffic,
        activation_traffic=activation_traffic,
        linear_compute_flops=linear_flops,
        attention_compute_flops=attention_flops,
        collective_bytes=b_remote_per_token_value * seq_len,
    )
    return TrafficSignature(
        phase="prefill",
        b_local_bytes=b_local,
        b_remote_bytes=b_remote_per_token_value * seq_len,
        n_messages=n_messages,
        compute_flops=linear_flops + attention_flops,
        token_count=seq_len * batch,
        local_footprint_bytes=local_footprint,
        weight_footprint_bytes=weight_footprint,
        kv_footprint_bytes=kv_footprint,
        activation_footprint_bytes=activation_footprint,
        weight_traffic_bytes=weight_traffic,
        kv_traffic_bytes=kv_traffic,
        activation_traffic_bytes_value=activation_traffic,
        semantic_stages=semantic_stages,
    )


def _allocate_stage_local_bytes(
    amounts: tuple[float, ...],
    local_fraction: float,
) -> tuple[float, ...]:
    """Apply one class placement fraction with a conservation residual."""
    local_total = math.fsum(amounts) * local_fraction
    positive_indices = [index for index, amount in enumerate(amounts) if amount > 0.0]
    if not positive_indices:
        return (0.0,) * len(amounts)

    last_positive = positive_indices[-1]
    assigned = []
    result = []
    for index, amount in enumerate(amounts):
        if index == last_positive:
            local_amount = local_total - math.fsum(assigned)
        else:
            local_amount = amount * local_fraction
            if amount > 0.0:
                assigned.append(local_amount)
        result.append(local_amount)
    return tuple(result)


def _reconcile_class_route_total(
    class_routes: tuple[tuple[float, ...], ...],
    target_total: float,
) -> tuple[tuple[float, ...], ...]:
    """Absorb cross-class rounding, preferring an exact aggregate.

    Each class is first split with its own placement fraction. Floating-point
    multiplication is not distributive, so those independently rounded class
    totals can differ by an ULP from the backward-compatible aggregate stream.
    Put only that numerical residual into the smallest positive routed item.
    Exact reconciliation remains the first choice and preserves the historical
    bit-exact path. Some large-target/small-stage combinations have no exact
    representable grouped sum and alternate between adjacent floats; after a
    detected cycle or the bounded correction attempts, accept only the closest
    visited state within one ULP of ``target_total``.
    """
    mutable = [list(route) for route in class_routes]

    def grouped_total() -> float:
        return math.fsum(
            mutable[0][index] + mutable[1][index] + mutable[2][index]
            for index in range(len(mutable[0]))
        )

    def frozen_routes() -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(route) for route in mutable)

    tolerance = math.ulp(target_total)
    best_routes = frozen_routes()
    best_error = math.inf
    visited: set[tuple[tuple[float, ...], ...]] = set()

    for _ in range(8):
        current_total = grouped_total()
        residual = target_total - current_total
        if residual == 0.0:
            return frozen_routes()

        current_routes = frozen_routes()
        current_error = abs(residual)
        if current_error < best_error:
            best_routes = current_routes
            best_error = current_error
        if current_routes in visited:
            break
        visited.add(current_routes)

        candidates = [
            (value, class_index, stage_index)
            for class_index, route in enumerate(mutable)
            for stage_index, value in enumerate(route)
            if value > 0.0 and value + residual >= 0.0
        ]
        if candidates:
            _, class_index, stage_index = min(candidates)
        elif residual > 0.0:
            class_index, stage_index = 0, len(mutable[0]) - 1
        else:
            raise RuntimeError("cannot reconcile a negative semantic route")

        value = mutable[class_index][stage_index]
        adjusted = value + residual
        if adjusted == value:
            adjusted = math.nextafter(
                value,
                math.inf if residual > 0.0 else -math.inf,
            )
        mutable[class_index][stage_index] = adjusted

    # Evaluate the state produced by the final bounded adjustment too.
    final_residual = target_total - grouped_total()
    if final_residual == 0.0:
        return frozen_routes()
    if abs(final_residual) < best_error:
        best_routes = frozen_routes()
        best_error = abs(final_residual)

    if best_error <= tolerance:
        return best_routes
    raise RuntimeError(
        "failed to conserve semantic route aggregate within target ULP: "
        f"best_error={best_error}, tolerance={tolerance}"
    )


def _build_stage_traffic_streams(
    stages: tuple[StageSignature, ...],
    placement: ClassPlacementFractions,
    side_fractions: tuple[float, float, float],
    r: float,
    local_total_bytes: float,
    side_total_bytes: float,
) -> tuple[StageTrafficStreams, ...]:
    """Apply class placement and route both complements without subtraction."""
    weight_amounts = tuple(stage.weight_traffic_bytes for stage in stages)
    kv_amounts = tuple(stage.kv_traffic_bytes for stage in stages)
    activation_amounts = tuple(stage.activation_traffic_bytes for stage in stages)
    collective_amounts = tuple(stage.collective_bytes for stage in stages)

    weight_local = _allocate_stage_local_bytes(
        weight_amounts, placement.weight_local
    )
    kv_local = _allocate_stage_local_bytes(kv_amounts, placement.kv_local)
    activation_local = _allocate_stage_local_bytes(
        activation_amounts, placement.activation_local
    )
    weight_side = _allocate_stage_local_bytes(
        weight_amounts, side_fractions[0]
    )
    kv_side = _allocate_stage_local_bytes(kv_amounts, side_fractions[1])
    activation_side = _allocate_stage_local_bytes(
        activation_amounts, side_fractions[2]
    )
    weight_local, kv_local, activation_local = _reconcile_class_route_total(
        (weight_local, kv_local, activation_local),
        local_total_bytes,
    )
    weight_side, kv_side, activation_side = _reconcile_class_route_total(
        (weight_side, kv_side, activation_side),
        side_total_bytes,
    )
    h2h = _allocate_stage_local_bytes(collective_amounts, 1.0 - r)
    gpu_direct = _allocate_stage_local_bytes(collective_amounts, r)

    return tuple(
        StageTrafficStreams(
            layer_index=stage.layer_index,
            sublayer=stage.sublayer,
            weight_local_bytes=weight_local[index],
            weight_side_bytes=weight_side[index],
            kv_local_bytes=kv_local[index],
            kv_side_bytes=kv_side[index],
            activation_local_bytes=activation_local[index],
            activation_side_bytes=activation_side[index],
            compute_flops=stage.compute_flops,
            h2h_bytes=h2h[index],
            gpu_direct_bytes=gpu_direct[index],
        )
        for index, stage in enumerate(stages)
    )


def split_traffic(
    signature: TrafficSignature,
    r_remote: float,
    r: float,
    *,
    local_hbm_placement_mode: str = "fixed_fraction",
    total_local_capacity_bytes: float | None = None,
    side_hbm_capacity_bytes: float | None = None,
) -> TrafficStreams:
    """Split local-memory and collective streams under the selected placement.

    ``fixed_fraction`` preserves the original ``r_remote`` behavior exactly.
    ``capacity_overflow`` ignores that input and derives placement from class
    footprints. The latter uses aggregate pooled local capacity. Per-stack
    placement feasibility remains a separate downstream check and may make an
    aggregate-feasible configuration infeasible under a non-uniform
    ``local_hbm_mapping_policy``.
    """
    if not 0.0 <= r <= 1.0:
        raise ValueError("r must be within [0, 1]")
    if local_hbm_placement_mode not in {"fixed_fraction", "capacity_overflow"}:
        raise ValueError(
            "local_hbm_placement_mode must be fixed_fraction or capacity_overflow"
        )

    resident_footprint = (
        signature.b_local_bytes
        if signature.local_footprint_bytes is None
        else signature.local_footprint_bytes
    )
    if local_hbm_placement_mode == "fixed_fraction":
        if not 0.0 <= r_remote <= 1.0:
            raise ValueError("r_remote must be within [0, 1]")
        local_hbm_bytes = (1.0 - r_remote) * signature.b_local_bytes
        side_hbm_bytes = r_remote * signature.b_local_bytes
        effective_r_remote = r_remote
        local_resident_footprint = (1.0 - r_remote) * resident_footprint
        side_resident_footprint = r_remote * resident_footprint
        class_placement = ClassPlacementFractions(
            weight_local=1.0 - r_remote,
            kv_local=1.0 - r_remote,
            activation_local=1.0 - r_remote,
        )
        # Preserve the user's remote fraction directly. Reconstructing it as
        # 1.0 - local_fraction would erase valid tiny positive inputs after
        # floating-point rounding.
        class_side_fractions = (r_remote, r_remote, r_remote)
    else:
        class_values = (
            (signature.weight_footprint_bytes, signature.weight_traffic_bytes),
            (signature.kv_footprint_bytes, signature.kv_traffic_bytes),
            (
                signature.activation_footprint_bytes,
                signature.activation_traffic_bytes_value,
            ),
        )
        if any(
            footprint is None or traffic is None
            for footprint, traffic in class_values
        ):
            raise ValueError(
                "capacity_overflow requires class-level footprint and traffic "
                "fields on TrafficSignature"
            )
        if total_local_capacity_bytes is None:
            raise ValueError(
                "capacity_overflow requires total_local_capacity_bytes"
            )

        # weight -> KV -> activation is a deterministic MVP modeling choice for
        # this round, NOT a literature-established optimal placement policy.
        # NUMA-Aware GPUs motivates modeling capacity-driven placement at all,
        # but does not prescribe this specific class ordering. A future round
        # could expose this as a configurable priority (for example,
        # placement_priority=("weight", "kv", "activation")) to study
        # placement-order sensitivity; that is out of scope here.
        remaining = total_local_capacity_bytes
        local_hbm_bytes = 0.0
        side_hbm_bytes = 0.0
        local_resident_footprint = 0.0
        side_resident_footprint = 0.0
        class_local_fractions = []
        class_side_fraction_values = []
        for class_footprint, class_traffic in class_values:
            class_local_footprint = min(
                class_footprint,
                max(remaining, 0.0),
            )
            class_local_fraction = (
                class_local_footprint / class_footprint
                if class_footprint > 0.0
                else 1.0
            )
            class_local_fractions.append(class_local_fraction)
            class_side_fraction_values.append(1.0 - class_local_fraction)
            remaining -= class_local_footprint
            local_resident_footprint += class_local_footprint
            side_resident_footprint += class_footprint - class_local_footprint
            # Apply each placement fraction only to its own class traffic.
            local_hbm_bytes += class_local_fraction * class_traffic
            side_hbm_bytes += (1.0 - class_local_fraction) * class_traffic

        # Reconstructing the used capacity across class boundaries can round
        # one ULP above the capacity supplied by the caller. The greedy fill
        # cannot physically place more than that capacity, so keep the exposed
        # resident total within its defining bound. Traffic remains the exact
        # class-by-class split computed above.
        local_resident_footprint = min(
            local_resident_footprint,
            total_local_capacity_bytes,
        )

        if (
            side_hbm_capacity_bytes is not None
            and side_resident_footprint > side_hbm_capacity_bytes
        ):
            raise SideHBMCapacityError(
                side_resident_footprint,
                side_hbm_capacity_bytes,
            )
        total_memory_traffic = local_hbm_bytes + side_hbm_bytes
        effective_r_remote = (
            side_hbm_bytes / total_memory_traffic
            if total_memory_traffic > 0.0
            else 0.0
        )
        class_placement = ClassPlacementFractions(
            weight_local=class_local_fractions[0],
            kv_local=class_local_fractions[1],
            activation_local=class_local_fractions[2],
        )
        class_side_fractions = (
            class_side_fraction_values[0],
            class_side_fraction_values[1],
            class_side_fraction_values[2],
        )

    semantic_stages = (
        _build_stage_traffic_streams(
            signature.semantic_stages,
            class_placement,
            class_side_fractions,
            r,
            local_hbm_bytes,
            side_hbm_bytes,
        )
        if signature.semantic_stages is not None
        else None
    )
    if semantic_stages is not None:
        n_side_requests = sum(
            stage.side_hbm_bytes > 0.0 for stage in semantic_stages
        )
    elif local_hbm_placement_mode == "fixed_fraction":
        # Preserve the legacy aggregate-signature condition bit-for-bit: its
        # request hint followed r_remote even for a zero-byte synthetic input.
        n_side_requests = signature.n_side_requests if r_remote > 0.0 else 0
    else:
        n_side_requests = (
            signature.n_side_requests if side_hbm_bytes > 0.0 else 0
        )

    return TrafficStreams(
        local_hbm_bytes=local_hbm_bytes,
        side_hbm_bytes=side_hbm_bytes,
        h2h_bytes=(1.0 - r) * signature.b_remote_bytes,
        gpu_direct_bytes=r * signature.b_remote_bytes,
        n_messages=signature.n_messages,
        n_side_requests=n_side_requests,
        effective_r_remote=effective_r_remote,
        local_resident_footprint_bytes=local_resident_footprint,
        side_resident_footprint_bytes=side_resident_footprint,
        class_placement=class_placement,
        semantic_stages=semantic_stages,
    )
