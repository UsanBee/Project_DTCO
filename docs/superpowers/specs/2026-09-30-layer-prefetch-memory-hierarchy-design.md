# Layer-Prefetch Memory Hierarchy Design

Date: 2026-09-30

Status: approved architecture design; implementation pending plan review

This design supersedes
`docs/superpowers/specs/2026-09-30-path-segment-accounting-design.md` as the
implementation target. The earlier path-accounting design remains useful as
the instrumentation foundation, but its path taxonomy is extended here for
the new hierarchy.

## 1. Research question

The simulator shall test whether an eight-lane uHBM staging tier can hide
layer-wise weight and KV movement behind the preceding transformer layer, and
whether the resulting GPU-facing bandwidth advantage outweighs its additional
prefetch and writeback work relative to conventional 2.5D HBM.

This is a Stage 2 byte-rate, capacity, latency, energy, and event-dependency
model. It does not model bank conflicts, address hashing, controller RTL,
signal integrity, PDN/IR drop, thermal fields, or EM behavior. Link parameters
remain direct scenario inputs until Stage 3 HFSS/ADS calibration.

## 2. Compatibility boundary

The existing Stage 2 simulator and all existing YAML files remain the legacy
execution mode. Their D-1 ring rounds, D-2 semantic stages, D-3 pillar bypass,
CSV schemas, timing, and results must remain unchanged.

The new hierarchy is selected only through a dedicated prefetch configuration
and entry point. It uses new focused modules rather than replacing the legacy
`split_traffic()` placement model. Existing legacy configs therefore require
no new fields and continue to run bit-exactly.

## 3. Nominal physical organizations

Both organizations contain eight symmetric lanes per GPU. Hardware parameters
are scalar per-lane values replicated across all eight lanes; v1 does not
model physical lane-to-lane variation.

### B0: conventional 2.5D

Each lane is:

```text
capacity_memory[g,i] -> capacity_link[g,i] -> hbm_2_5d[g,i] -> GPU[g]
```

- Eight conventional 16-Hi HBM packages per GPU.
- Nominal capacity is 48 GB per package, 384 GB per GPU.
- Full historical KV state is resident in conventional HBM.
- Conventional HBM also holds current/next weight staging buffers when
  weights are streamed.
- New KV is written directly from the GPU into conventional HBM.
- There is no separate side-HBM tier and no default capacity-memory-to-GPU
  direct-access path.

### B2: uHBM

Each lane is:

```text
capacity_memory[g,i] -> capacity_link[g,i] --------> uHBM[g,i] -> GPU[g]
side_hbm[g,i] <---- shared bidirectional side_link[g,i] ----> uHBM[g,i]
```

- Eight 4-Hi uHBM packages per GPU, nominally 12 GB each and 96 GB total.
- Eight 16-Hi side-HBM packages per GPU, nominally 48 GB each and 384 GB
  total.
- Weight backing state is stored in the eight capacity memories.
- Full historical KV state is stored in the eight side HBMs.
- uHBM holds current/next layer staging, scratch, in-flight/writeback state,
  and optionally the persistent TP-sharded weight set.
- Capacity memory connects directly to its paired uHBM; data does not traverse
  side HBM.
- Each uHBM may use only its paired side HBM and capacity memory.

The 12 GB value is the nominal capacity assumption obtained by linearly
scaling a 48 GB 16-Hi package to 4-Hi. It is a configurable Stage 2 capacity,
not a claim that stacking has no ECC, spare-die, PHY, or yield overhead.

### Native and capacity controls

The primary comparison keeps the native capacities above. Side-HBM capacity
is not artificially reduced. An optional synthetic capacity-equalized control
may increase B0 capacity to match B2's aggregate HBM capacity, but it is
reported separately and never presented as a product baseline.

## 4. Data placement

TP shards weight and KV state across GPUs using the existing Llama-2 model
assumptions. Each GPU's shard is then uniformly striped across its eight
lanes:

- one eighth of every layer's weight backing state per capacity lane;
- one eighth of historical KV per side-HBM lane in B2;
- one eighth of historical KV per conventional-HBM lane in B0; and
- one eighth of each newly generated KV increment per lane.

Layer execution requires all eight lanes. The layer-ready time is the maximum
weight/KV ready time across lanes. Optional declarative weighted mappings may
be added for workload-placement sensitivity, but the nominal hardware remains
physically symmetric. No cross-lane borrowing or routing is modeled.

Capacity memory stores weight only. Side HBM stores KV only. If side HBM (B2)
or conventional HBM's mandatory KV allocation (B0) is exhausted, the run is
infeasible; v1 does not invent eviction, compression, paging, or KV spill to
capacity memory.

## 5. Weight residency policies

Two explicit policies are supported.

### `persistent_if_fit`

Weights are assumed preloaded before the measured inference interval when the
entire per-GPU TP shard fits after mandatory runtime state is reserved. Model
loading time is not included in TTFT/TPOT.

For B2, mandatory uHBM state contains current/next KV staging, collective
scratch, and in-flight/writeback buffers. For B0, mandatory conventional-HBM
state contains the full historical KV state, scratch, and KV growth. If the
complete weight shard does not fit, the policy falls back atomically to
layer-wise streaming; v1 does not implement partial layer pinning or LRU.

### `streaming`

Weight backing remains in capacity memory. Current and next layer weights use
double buffers and are re-streamed for every decode token. This policy exposes
the upstream bandwidth needed to sustain the uHBM execution rate.

If the mandatory current/next buffers do not fit, the architecture point is
infeasible.

## 6. Workload phases and state

### Fresh-prompt prefill

- No historical KV is prefetched.
- Layer weights are resident or prefetched according to the weight policy.
- Each attention layer generates prompt KV.
- B2 writes generated prompt KV through uHBM and the paired side link into
  side HBM; B0 writes it directly into conventional HBM.
- Prefix-cache reuse and prior-session KV are outside v1.

### Decode

`decode_tokens` is a sweepable positive integer. For each token and layer:

- streaming weights move from capacity memory to local execution memory;
- B2 historical KV moves from side HBM to uHBM;
- B0 historical KV is already resident in conventional HBM;
- attention generates one token's new KV; and
- the new KV is committed before the next token reads that layer's KV.

KV resident capacity grows across decoded tokens. The event graph carries
writeback completion and resident state from one token to the next.

Supported run modes are `prefill_only`, `decode_only`, and `prefill_decode`.
`decode_only` starts with the configured sequence-length KV already resident
in its backing tier.

### Cold and steady measurements

- `cold_start` begins with empty current/next staging buffers. Layer 0
  preparation is measured.
- `steady_state` executes an unmeasured warm-up decode token and reports the
  following configured tokens while preserving the actual buffer/writeback
  state. It does not use the causally invalid relaxed collective DAG.

Persistent model-load time remains outside both measurements.

## 7. Semantic layer schedule

D-2 attention/MLP semantic stages remain the intra-layer schedule. Prefetch
operates at full-transformer-layer granularity:

1. layer `i` becomes ready only after every required weight and historical-KV
   chunk is present in all eight local lanes;
2. when layer `i` starts, prefetch for layer `i+1` is launched;
3. that prefetch overlaps attention, its collective, MLP, and its collective;
4. layer `i+1` compute waits for both its data and the preceding collective;
5. attention-generated KV writeback may overlap the remaining layer work, but
   must complete before the next token's corresponding KV prefetch.

Transfers are chunked and pipelined, but v1 uses a full-layer-ready barrier.
GPU compute never consumes a partially arrived layer. Intra-layer tensor/tile
streaming requires kernel consumption order and is deferred to Stage 4 trace
calibration.

For layer `i`:

```text
layer_ready_i = max(weight_ready_i, kv_ready_i) over all eight lanes
prefetch_slack_i = compute_end_(i-1) - layer_ready_i
prefetch_stall_i = max(0, -prefetch_slack_i)
```

## 8. Chunked DMA and resource semantics

The three flows are:

```text
weight:       capacity read -> capacity link -> local-memory write
KV prefetch:  side-HBM read -> shared side link -> uHBM write
KV writeback: uHBM read -> shared side link -> side-HBM write
```

Weight, KV-prefetch, and KV-writeback chunk sizes are independent config
values. A residual-corrected split conserves each transfer's byte total.

`max_inflight_chunks_per_lane` bounds the shared DMA descriptor pool in a
lane. A value of one is the conservative no-inter-chunk-pipeline bound; a
finite larger value permits actual source/link/destination pipeline overlap;
`null` is the optimistic all-enqueued bound. It is a sensitivity parameter,
not a calibrated hardware constant.

Every physical resource is non-preemptive FCFS at chunk granularity:

- one capacity-memory read resource per lane;
- one capacity link per lane;
- one shared bidirectional side link per B2 lane;
- one shared side-HBM datapath per B2 lane for reads and writes;
- one shared local-memory datapath per lane for GPU reads, GPU writes,
  prefetch writes, and writeback reads; and
- existing per-GPU compute and D-1 collective resources.

Capacity and side links are independent and may run concurrently, but they
converge on the shared uHBM datapath. KV prefetch and writeback compete on the
same side link and side-HBM datapath. There is no fixed directional bandwidth
partition and no priority; simultaneous-arrival ties use a documented stable
ordering only for reproducibility.

The eight paired prefetch/writeback fabrics are separate from the D-1
inter-GPU collective H2H fabric. Only TP collectives use D-1 H2H/GPU-direct
resources.

## 9. Capacity-memory profiles

The core uses technology-neutral direct parameters:

- capacity per lane;
- memory read bandwidth, latency, and dynamic pJ/bit;
- capacity-link bandwidth, latency, and dynamic pJ/bit; and
- optional distance metadata.

Memory service and link transport are separate pipeline resources. Distance
metadata does not imply a fabricated distance-to-bandwidth formula. Named
DDR/LPDDR/CXL-attached presets are added only after cited calibration;
`conservative`, `nominal`, and `aggressive` scenario profiles remain clearly
labeled placeholders until then.

## 10. Path accounting and energy

`RateResource` traffic tags provide bytes, requests, queue time, service time,
and fixed-latency time without changing event order or legacy totals. New
classes include:

- capacity-memory read;
- capacity-to-uHBM transport;
- capacity-to-2.5D-HBM transport;
- side-HBM read and write;
- side-to-uHBM and uHBM-to-side transport;
- uHBM GPU read/write and prefetch/writeback accesses;
- 2.5D-HBM GPU read/write and prefetch accesses; and
- existing collective H2H/GPU-direct traffic.

Dynamic energy is actual transferred bytes times the configured pJ/bit for
each service/transport segment. A byte is charged once per physical segment
it traverses. Existing local-HBM idle energy remains system-wide. New tier
idle-power terms are not invented without sourced parameters.

Resource-time sums are accounting totals and may exceed elapsed time because
resources overlap. They are not critical-path exposure.

## 11. Results

Summary sweep rows report:

- architecture, run mode, start state, decode tokens, weight policy, chunk
  sizes, and in-flight limit;
- cold TTFT, first-token latency, token TPOT series summary, steady mean,
  median, maximum, and throughput;
- total/weight/KV prefetch stall, hidden-prefetch fraction, deadline misses,
  and the limiting class;
- KV-writeback deadline misses and maximum lateness;
- capacity state, residency decision, first infeasible token, and reason;
- per-path bytes, requests, queue/service/latency time, utilization, and
  dynamic energy;
- lane mean/max utilization and critical lane; and
- existing collective timing/energy metrics.

Token/layer/lane/chunk details are available through an optional in-memory
trace for selected points. They are not expanded into the Cartesian sweep
CSV by default.

## 12. Validation invariants

Tests must establish:

1. exact byte conservation across lanes, chunks, and paths;
2. eight-lane symmetry under uniform mapping;
3. full-layer-ready barriers across all lanes;
4. actual chunk pipeline behavior and bounded in-flight depth;
5. FCFS contention between KV prefetch/writeback and among uHBM access types;
6. no prefetch traffic on D-1 collective H2H resources;
7. B0 historical KV residency with no KV-prefetch transport;
8. B2 KV prefetch and writeback deadlines across tokens;
9. correct `persistent_if_fit` versus streaming fallback and infeasibility;
10. cold-start layer-0 cost and warm-up-excluded steady-state metrics;
11. fresh prefill KV creation and end-to-end prefill/decode state transfer;
12. per-path energy conservation without crossing double counting; and
13. bit-exact legacy config, result, and test behavior.

Analytical toy cases with unit rates pin exact timings. Nominal research
profiles are not accepted as validated hardware values until Stage 3/4
calibration.

## 13. Out of scope

- intra-layer tensor/tile streaming;
- partial weight pinning, caching, LRU, or runtime migration;
- prefix-cache reuse or multi-session scheduling;
- KV eviction, compression, sparse attention, or paging to capacity memory;
- actual address mapping, bank/channel conflicts, or controller arbitration;
- capacity-memory direct GPU access as the primary B0 baseline;
- fixed read/write bandwidth partitions or priority scheduling;
- cross-lane borrowing or asymmetric physical lane parameters;
- deriving link parameters from distance or signal/power TSV counts;
- PDN, IR-drop, thermal-field, SI, EM, or fabrication feasibility simulation;
- pipeline parallelism beyond the existing PP=1 assumption; and
- claiming placeholder capacity/link profiles as measured uHBM values.
