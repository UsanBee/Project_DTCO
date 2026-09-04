"""Workload-layer traffic signatures derived from the Stage 1 model."""

from __future__ import annotations

from dataclasses import dataclass

from stage1_model import MODELS, b_local_per_token, b_remote_per_token


@dataclass(frozen=True)
class TrafficSignature:
    phase: str
    b_local_bytes: float
    b_remote_bytes: float
    n_messages: int
    compute_flops: float
    token_count: int = 1
    n_side_requests: int = 1


@dataclass(frozen=True)
class TrafficStreams:
    local_hbm_bytes: float
    side_hbm_bytes: float
    h2h_bytes: float
    gpu_direct_bytes: float
    n_messages: int
    n_side_requests: int


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
    b_local = b_local_per_token(model, tp, seq_len, batch)
    b_remote, n_messages = b_remote_per_token(model, tp, batch)
    # TODO(태현): 실제 kernel efficiency를 반영한 FLOP model로 교체한다.
    compute_flops = 2.0 * model["n_params"] * batch / tp
    return TrafficSignature(
        phase="decode",
        b_local_bytes=b_local,
        b_remote_bytes=b_remote,
        n_messages=n_messages,
        compute_flops=compute_flops,
        n_side_requests=model["n_layers"],
    )


def build_prefill_signature(
    model_name: str,
    tp: int,
    seq_len: int,
    batch: int = 1,
) -> TrafficSignature:
    """Build a compute-dominant prefill signature for a full sequence."""
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name}")
    if tp < 1 or seq_len < 1 or batch < 1:
        raise ValueError("tp, seq_len, and batch must be positive")

    model = MODELS[model_name]
    # TODO(태현): Stage 3/4 data로 prefill memory traffic approximation을 보정한다.
    b_local = b_local_per_token(model, tp, seq_len, batch)
    b_remote_per_token_value, n_messages = b_remote_per_token(model, tp, batch)
    return TrafficSignature(
        phase="prefill",
        b_local_bytes=b_local,
        b_remote_bytes=b_remote_per_token_value * seq_len,
        n_messages=n_messages,
        compute_flops=2.0 * model["n_params"] * seq_len * batch / tp,
        token_count=seq_len * batch,
        n_side_requests=model["n_layers"],
    )


def split_traffic(
    signature: TrafficSignature,
    r_remote: float,
    r: float,
) -> TrafficStreams:
    """Split local-memory and collective streams with independent fractions."""
    if not 0.0 <= r_remote <= 1.0:
        raise ValueError("r_remote must be within [0, 1]")
    if not 0.0 <= r <= 1.0:
        raise ValueError("r must be within [0, 1]")

    return TrafficStreams(
        local_hbm_bytes=(1.0 - r_remote) * signature.b_local_bytes,
        side_hbm_bytes=r_remote * signature.b_local_bytes,
        h2h_bytes=(1.0 - r) * signature.b_remote_bytes,
        gpu_direct_bytes=r * signature.b_remote_bytes,
        n_messages=signature.n_messages,
        n_side_requests=signature.n_side_requests if r_remote > 0.0 else 0,
    )
