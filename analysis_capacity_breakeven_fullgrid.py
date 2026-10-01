"""Full-grid capacity breakeven analysis (post C-1/C-2 fixes).

Extends the earlier single-point capacity-breakeven experiment
(analysis_capacity_breakeven_v2.py, decode/prefill x tp={4,8} x
seq_len=4096 x batch=1 only) to the project's established full workload
grid (tp x {2,4,8}, seq_len x {512,4096,16384}, batch x {1,4,8,16,32}),
using the now-fixed materialized-mode headroom reservation (C-1) and both
event-driven local/side memory concurrency modes (C-2: full_overlap,
serialized) as MODEL-LEVEL bounding cases -- not as a claimed guarantee on
real hardware. No simulator source changes; analysis-only script.

Checkpointed / resumable: each device shell invocation has a short wall
clock budget, so this processes workloads in chunks (--chunk N, default
all remaining) and records progress in results/fullgrid_checkpoint.json.
Re-running the script continues from the next unprocessed workload;
partial summary/detail CSVs are appended to (headers written once).

For each workload W=(phase,tp,seq_len,batch) this computes, per concurrency
mode:
  - min_feasible_capacity_gb: smallest tested local-HBM capacity at which
    B2 (uHBM, capacity_overflow placement, materialized comm) is feasible
    at all (LocalHBMCapacityError/SideHBMCapacityError-free).
  - breakeven_capacity_gb: smallest tested capacity beyond which B2's
    metric (TPOT for decode, TTFT for prefill) never exceeds B0's metric
    again for any larger tested capacity. This definition is written to be
    robust to the known non-monotonic ("U-shaped") full_overlap curves: a
    naive "first capacity where metric<=B0" scan can pick a transient dip
    that regresses again at a larger capacity; scanning from the top down
    and stopping at the first regression avoids that trap.

The full_overlap vs serialized gap at each workload is reported as a
concurrency_gap_gb column -- interpreted as a MODEL-LEVEL sensitivity
envelope width, not as a bound guaranteed to contain real hardware
behavior (see README_stage2.md's C-2 documentation of these as two
event-driven bounding cases internal to this simulator).
"""
import sys
sys.path.insert(0, ".")
import csv
import json
import os

from uhbm_simulator.simulator import (
    SimulationParameters,
    run_simulation,
    LocalHBMCapacityError,
    SideHBMCapacityError,
)
from uhbm_simulator.workload import (
    build_decode_signature,
    build_prefill_signature,
    split_traffic,
)

MODEL_NAME = "Llama-70B"
GPU_PEAK_FLOPS = 990e12
BASE_DIE_LATENCY_S = 5e-9
H2H_BW_BPS = 336e9
H2H_LATENCY_S = 20e-9
UHBM_GPU_DIRECT_BW_BPS = 768e9
UHBM_GPU_DIRECT_LATENCY_S = 20e-9
NVLINK_BW_BPS = 900e9
NVLINK_LATENCY_S = 9e-6
CONV_HBM_STACK_COUNT = 5
CONV_HBM_STACK_BW_BPS = 3.35e12 / 5
CONV_HBM_STACK_CAPACITY_BYTES = 80e9 / 5
CONV_HBM_LATENCY_S = 106.7e-9
UHBM_STACK_COUNT = 4
UHBM_STACK_BW_BPS = 1.2e12

COMMON = dict(
    thermal_scenario="no_throttling",
    local_hbm_mapping_policy="uniform", local_hbm_mapping_weights=None,
    side_hbm_bw_bps=4.8e12, side_hbm_latency_s=50e-9,
    gpu_peak_flops=GPU_PEAK_FLOPS, base_die_latency_s=BASE_DIE_LATENCY_S,
    comm_mode="materialized",
)

SUMMARY_CSV = "results/capacity_breakeven_fullgrid_summary.csv"
DETAIL_CSV = "results/capacity_breakeven_fullgrid_detail.csv"
CHECKPOINT = "results/fullgrid_checkpoint.json"


def b0_params(tp):
    return SimulationParameters(
        **COMMON, tp=tp, r=1.0, r_remote=0.0,
        local_hbm_stack_count=CONV_HBM_STACK_COUNT, local_hbm_stack_bw_bps=CONV_HBM_STACK_BW_BPS,
        local_hbm_stack_capacity_bytes=CONV_HBM_STACK_CAPACITY_BYTES, local_hbm_latency_s=CONV_HBM_LATENCY_S,
        h2h_bw_bps=H2H_BW_BPS, h2h_latency_s=H2H_LATENCY_S,
        gpu_direct_bw_bps=NVLINK_BW_BPS, gpu_direct_latency_s=NVLINK_LATENCY_S,
    )


def b2_params(tp, total_capacity_bytes, concurrency):
    return SimulationParameters(
        **COMMON, tp=tp, r=0.0, r_remote=0.0,
        local_hbm_stack_count=UHBM_STACK_COUNT, local_hbm_stack_bw_bps=UHBM_STACK_BW_BPS,
        local_hbm_stack_capacity_bytes=total_capacity_bytes / UHBM_STACK_COUNT, local_hbm_latency_s=0.0,
        h2h_bw_bps=H2H_BW_BPS, h2h_latency_s=H2H_LATENCY_S,
        gpu_direct_bw_bps=UHBM_GPU_DIRECT_BW_BPS, gpu_direct_latency_s=UHBM_GPU_DIRECT_LATENCY_S,
        local_hbm_placement_mode="capacity_overflow",
        local_side_memory_concurrency=concurrency,
    )


def metric_of(result, phase):
    return result.tpot_s if phase == "decode" else result.ttft_s


def run_b2(signature, phase, tp, total_capacity_bytes, concurrency):
    params = b2_params(tp, total_capacity_bytes, concurrency)
    try:
        result = run_simulation(signature, params)
    except (LocalHBMCapacityError, SideHBMCapacityError):
        return None, None, False
    streams = split_traffic(
        signature, params.r_remote, params.r,
        local_hbm_placement_mode=params.local_hbm_placement_mode,
        total_local_capacity_bytes=total_capacity_bytes,
    )
    return metric_of(result, phase), streams.effective_r_remote, True


CAPACITY_MULTIPLIERS = [
    0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1,
    1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0,
]


def capacity_grid_for(footprint_bytes):
    return [footprint_bytes * m for m in CAPACITY_MULTIPLIERS]


def find_min_feasible_and_breakeven(caps_bytes, metrics, feas, b0_metric):
    min_feasible_gb = None
    for c, f in zip(caps_bytes, feas):
        if f:
            min_feasible_gb = c / 1e9
            break
    breakeven_gb = None
    for c, m, f in zip(reversed(caps_bytes), reversed(metrics), reversed(feas)):
        if not f or m is None:
            break
        if m <= b0_metric:
            breakeven_gb = c / 1e9
        else:
            break
    return min_feasible_gb, breakeven_gb


def all_workloads():
    tps = [2, 4, 8]
    seq_lens = [512, 4096, 16384]
    batches = [1, 4, 8, 16, 32]
    phases = [("decode", build_decode_signature), ("prefill", build_prefill_signature)]
    out = []
    for phase, builder in phases:
        for tp in tps:
            for seq_len in seq_lens:
                for batch in batches:
                    out.append((phase, builder, tp, seq_len, batch))
    return out


def process_one(phase, builder, tp, seq_len, batch):
    sig = builder(MODEL_NAME, tp, seq_len, batch)

    try:
        b0_result = run_simulation(sig, b0_params(tp))
        b0_metric = metric_of(b0_result, phase)
        b0_feasible = True
    except (LocalHBMCapacityError, SideHBMCapacityError):
        b0_metric = None
        b0_feasible = False

    footprint = sig.local_footprint_bytes if sig.local_footprint_bytes is not None else sig.b_local_bytes
    caps = capacity_grid_for(footprint)

    summary = dict(
        phase=phase, tp=tp, seq_len=seq_len, batch=batch,
        footprint_gb=footprint / 1e9,
        b0_feasible=b0_feasible, b0_metric_s=b0_metric,
    )
    detail_rows = []

    for concurrency in ("full_overlap", "serialized"):
        metrics, feas = [], []
        for c in caps:
            m, eff_r, ok = run_b2(sig, phase, tp, c, concurrency)
            metrics.append(m)
            feas.append(ok)
            detail_rows.append(dict(
                phase=phase, tp=tp, seq_len=seq_len, batch=batch,
                concurrency=concurrency, capacity_gb=c / 1e9,
                feasible=ok, metric_s=m, effective_r_remote=eff_r,
                b0_metric_s=b0_metric,
                ratio_vs_b0=(m / b0_metric) if (m is not None and b0_metric) else None,
            ))
        if b0_feasible:
            min_feas_gb, breakeven_gb = find_min_feasible_and_breakeven(caps, metrics, feas, b0_metric)
        else:
            min_feas_gb, breakeven_gb = None, None
        summary[f"min_feasible_capacity_gb_{concurrency}"] = min_feas_gb
        summary[f"breakeven_capacity_gb_{concurrency}"] = breakeven_gb
        last_m = metrics[-1]
        summary[f"metric_s_at_max_capacity_{concurrency}"] = last_m
        summary[f"ratio_vs_b0_at_max_capacity_{concurrency}"] = (
            (last_m / b0_metric) if (last_m is not None and b0_metric) else None
        )

    be_full = summary.get("breakeven_capacity_gb_full_overlap")
    be_ser = summary.get("breakeven_capacity_gb_serialized")
    summary["concurrency_gap_gb"] = (
        (be_ser - be_full) if (be_full is not None and be_ser is not None) else None
    )
    return summary, detail_rows


SUMMARY_FIELDS = [
    "phase", "tp", "seq_len", "batch", "footprint_gb", "b0_feasible", "b0_metric_s",
    "min_feasible_capacity_gb_full_overlap", "breakeven_capacity_gb_full_overlap",
    "metric_s_at_max_capacity_full_overlap", "ratio_vs_b0_at_max_capacity_full_overlap",
    "min_feasible_capacity_gb_serialized", "breakeven_capacity_gb_serialized",
    "metric_s_at_max_capacity_serialized", "ratio_vs_b0_at_max_capacity_serialized",
    "concurrency_gap_gb",
]
DETAIL_FIELDS = [
    "phase", "tp", "seq_len", "batch", "concurrency", "capacity_gb",
    "feasible", "metric_s", "effective_r_remote", "b0_metric_s", "ratio_vs_b0",
]


def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"next_index": 0}


def save_checkpoint(next_index):
    with open(CHECKPOINT, "w") as f:
        json.dump({"next_index": next_index}, f)


def main():
    chunk = None
    for arg in sys.argv[1:]:
        if arg.startswith("--chunk="):
            chunk = int(arg.split("=", 1)[1])

    workloads = all_workloads()
    total = len(workloads)
    ckpt = load_checkpoint()
    start = ckpt["next_index"]

    if start >= total:
        print(f"already complete: {total}/{total} workloads processed")
        return

    end = total if chunk is None else min(total, start + chunk)

    summary_is_new = not os.path.exists(SUMMARY_CSV)
    detail_is_new = not os.path.exists(DETAIL_CSV)
    summary_f = open(SUMMARY_CSV, "a", newline="")
    detail_f = open(DETAIL_CSV, "a", newline="")
    summary_w = csv.DictWriter(summary_f, fieldnames=SUMMARY_FIELDS)
    detail_w = csv.DictWriter(detail_f, fieldnames=DETAIL_FIELDS)
    if summary_is_new:
        summary_w.writeheader()
    if detail_is_new:
        detail_w.writeheader()

    for i in range(start, end):
        phase, builder, tp, seq_len, batch = workloads[i]
        summary, detail_rows = process_one(phase, builder, tp, seq_len, batch)
        summary_w.writerow(summary)
        for row in detail_rows:
            detail_w.writerow(row)
        summary_f.flush()
        detail_f.flush()
        save_checkpoint(i + 1)
        print(f"progress: {i + 1}/{total} workloads "
              f"(phase={phase} tp={tp} seq_len={seq_len} batch={batch})", flush=True)

    summary_f.close()
    detail_f.close()
    print(f"chunk done: {end}/{total} workloads processed so far")


if __name__ == "__main__":
    main()
