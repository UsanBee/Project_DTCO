# Path-Segment Accounting Design

Date: 2026-09-30

## Purpose

Add traffic-source-aware accounting to the Stage 2 uHBM simulator so that a
single run can report separately:

- GPU workload access to the GPU's own local HBM;
- collective endpoint accesses to local HBM;
- the remote HBM datapath service for side-HBM reads;
- side-HBM payload transport over shared H2H, pillar bypass, or the
  independent comparison path; and
- collective transport over H2H or GPU-direct.

This is instrumentation only. It must not change resource construction,
request ordering, service duration, dependency ordering, capacity placement,
TTFT/TPOT, or existing aggregate energy.

## Scope and interpretation

The simulator's `local_hbm` resource remains an aggregate HBM-stack access
service. A `local_workload_read` observation therefore represents the modeled
GPU-to-local-HBM access, not a newly introduced or separately calibrated
vertical PHY. Similarly, a side path is decomposed into remote-HBM datapath
service plus its modeled transport resource; no new endpoint or controller
latency is invented.

The feature reports resource-accounting totals. Summed queue, service, and
fixed-latency times can exceed `elapsed_s` because requests on different
tiles, stacks, and links can overlap. They are not critical-path attribution
and must not be described as exposed time.

## Rejected alternatives

### Call-site byte counters only

Keeping counters only in simulator call sites would separate bytes but could
not attribute queue time on a shared H2H edge to collective traffic versus
side-HBM traffic. This is insufficient for the requested path analysis.

### Full event trace

Recording every event with timestamps would support arbitrary post-processing,
but it would substantially increase result volume and runtime. Per-event trace
export is outside this version's scope.

## Traffic classes

Production transfer requests use one of these fixed traffic-class strings:

| Traffic class | Meaning |
| --- | --- |
| `local_workload_read` | Weight, KV, or activation traffic served by the requester's local HBM |
| `collective_source_read` | Local-HBM source read before materialized or cut-through collective transport |
| `collective_final_write` | Materialized collective result written to the requester's local HBM |
| `collective_final_reread` | Materialized collective result reread from the requester's local HBM |
| `side_hbm_datapath_read` | Data service performed by the remote owner's side-HBM resource |
| `side_shared_h2h_transport` | Side-read data payload traversing shared H2H edges |
| `side_pillar_transport` | Side-read data payload traversing a directed pillar link |
| `side_independent_transport` | Logical independent-path crossing used for bytes and crossing energy only |
| `collective_h2h_transport` | Collective payload traversing H2H ring edges |
| `collective_gpu_direct_transport` | Collective payload traversing the GPU-direct resource |
| `capacity_memory_read` | Weight bytes read from a paired capacity-memory device |
| `capacity_to_uhbm` | Weight payload transported directly to its paired uHBM lane |
| `capacity_to_2_5d_hbm` | Weight payload transported to its paired conventional HBM lane |
| `side_hbm_read` / `side_hbm_write` | Historical-KV read or new-KV write in the paired side HBM |
| `side_to_uhbm` / `uhbm_to_side` | Bidirectional KV payload on the shared paired side link |
| `uhbm_gpu_read` / `gpu_uhbm_write` | GPU access to a B2 local uHBM lane |
| `uhbm_prefetch_write` / `uhbm_writeback_read` | B2 local staging endpoint accesses |
| `hbm_2_5d_gpu_read` / `gpu_hbm_2_5d_write` | GPU access to a B0 conventional-HBM lane |
| `hbm_2_5d_prefetch_write` | B0 local weight-prefetch endpoint access |
| `unclassified` | Compatibility fallback for direct/custom `RateResource.use(amount)` calls |

The names are constants in the resource/accounting layer rather than repeated
ad-hoc string literals. GPU compute is not a transfer path and remains outside
this classification.

## Resource-level accounting

`RateResource.use()` gains an optional traffic-class argument:

```python
resource.use(amount, traffic_class="local_workload_read")
```

The default is `unclassified`, preserving existing custom callers. Each
resource continues to update its existing total `ResourceStats` and also
updates a per-class stats entry from the same queue acquisition and service
event. Per-class statistics contain:

- resource request count;
- transferred amount;
- total queue time;
- total service time;
- total fixed-latency time; and
- first service start time.

The total fields remain the source of existing metrics. Tagged statistics are
additional observations; totals are not recomputed from tags, avoiding changes
in floating-point accumulation order.

The immutable result snapshot adds a `by_traffic_class` mapping to every
`ResourceMetric`. Existing top-level fields retain their names and values.

Example detailed query:

```python
metric = result.resource_metrics["h2h_link[1,2]"]
collective = metric.by_traffic_class["collective_h2h_transport"]
side = metric.by_traffic_class["side_shared_h2h_transport"]
```

The dedicated layer-prefetch engine exposes the same tagged observations
through `calculate_prefetch_metrics(config, result).paths`. Its summary CSV
stores a deterministic JSON mapping in `path_metrics_json`; it does not emit a
second per-lane or per-chunk CSV. Capacity-device service, capacity-link
transport, local-memory endpoint service, side-HBM service, paired side-link
transport, and D-1 collective transport remain separate accounting domains.

## Run-level path metrics

`SimulationResult` adds `path_metrics: dict[str, PathMetric]`. A `PathMetric`
contains:

- `resource_request_count`;
- `transferred_bytes`;
- `total_queue_time_s`;
- `total_service_time_s`;
- `total_latency_time_s`; and
- `timing_modeled`.

For physical resource-backed paths, metrics are aggregated from all resources'
tagged snapshots. Transport bytes mean physical edge-bytes for multi-edge H2H
routes, matching the current `collective_h2h_edge_bytes` and
`side_hbm_read_h2h_edge_bytes` semantics.

The independent side path deliberately has no transport `RateResource` in the
current model. Its `side_independent_transport` metric therefore exposes the
logical crossing bytes, sets `timing_modeled=False`, and leaves queue/service/
latency values absent rather than reporting a misleading zero-duration
transport. Its request count is derived from the corresponding side-HBM read
requests.

All defined path keys are present in a production result, including zero-valued
inactive physical paths. The compatibility `unclassified` key is included only
when unclassified transfer requests occurred.

## Energy attribution

Existing channel totals remain unchanged and continue to be computed from the
existing total resource metrics. A new path-energy mapping attributes dynamic
energy using tagged transferred bytes:

- local-HBM pJ/bit for local workload and each collective endpoint class;
- side-HBM datapath pJ/bit for `side_hbm_datapath_read`;
- H2H pJ/bit for collective and shared-side H2H classes;
- pillar pJ/bit for pillar side transport;
- independent crossing pJ/bit for the logical independent transport; and
- GPU-direct pJ/bit for direct collective transport.

Local-HBM idle/background energy is not assigned to a traffic path. The
following conservation relationships must hold within floating-point
tolerance:

```text
local_hbm_energy_j
  = local_workload_read
  + collective_source_read
  + collective_final_write
  + collective_final_reread

h2h_energy_j
  = collective_h2h_transport
  + side_shared_h2h_transport

dynamic path energy + local_hbm_idle_energy_j = total_energy_j
```

The legacy aggregate totals are calculated by their existing code path so that
successful pre-change design points remain bit-exact.

## Sweep CSV schema

The standard sweep CSV adds only aggregate path groups. Tile/stack/edge detail
stays available through `SimulationResult.resource_metrics` and is not emitted
to a second CSV.

The groups are:

- `local_workload`;
- `collective_local_endpoint` (source read + final write + final reread);
- `side_hbm_datapath`;
- `side_transport` (the selected shared-H2H, pillar, or independent route);
- `collective_transport` (H2H + GPU-direct).

For each group the CSV exposes:

```text
<group>_bytes
<group>_resource_request_count
<group>_total_queue_time_s
<group>_total_service_time_s
<group>_total_latency_time_s
<group>_energy_j
```

For `side_transport` in independent mode, the three timing fields are `None`
because transport timing is not modeled. Capacity-infeasible rows follow the
existing convention: all new simulation-result fields are `None` while
`capacity_feasible=False` remains populated.

These time columns are resource-accounting sums, not path wall-clock duration
or critical-path exposure. `README_stage2.md` will state that explicitly.

## Compatibility and invariants

- `RateResource.use(amount)` remains valid and records `unclassified` stats.
- Existing `ResourceMetric` total fields retain their values.
- Existing `SimulationResult` and `SweepRow` fields retain their names and
  semantics.
- Existing collective/side/pillar byte counters remain present.
- No new resource or latency is introduced.
- No event is added to or removed from the SimPy dependency graph.
- Existing TTFT, TPOT, communication time, overlap, capacity feasibility,
  utilization, channel energy, and total energy remain bit-exact for feasible
  points.

## Error handling

Production code must use a declared traffic class. An unknown explicit class
raises `ValueError` rather than silently creating an unreportable category.
The default compatibility call maps to the declared `unclassified` class.

When a production resource contains unexpected unclassified transfer traffic,
the detail remains observable in `resource_metrics`; tests ensure the standard
simulator paths are fully classified. Custom callers are not rejected.

## Tests

Tests will cover:

1. direct `RateResource.use(amount)` compatibility and explicit-tag stats;
2. per-resource tag bytes, requests, service, and latency summing to totals;
3. queue time being charged to the request class that actually waited;
4. shared H2H collective plus side-tag bytes equaling total H2H bytes;
5. materialized source-read, final-write, and final-reread separation;
6. cut-through source read without final write/reread;
7. pillar and shared-H2H side transports being mutually exclusive;
8. independent side transport exposing bytes/energy with unmodeled timing;
9. path-energy conservation against existing channel totals;
10. aggregate sweep columns and capacity-infeasible `None` behavior;
11. no unclassified production transfer requests; and
12. bit-exact regression for existing timing, capacity, byte, and aggregate
    energy results.

## Documentation

`README_stage2.md` will document the traffic classes, detailed programmatic
query example, aggregate CSV groups, independent-path timing limitation, and
the distinction between accumulated resource time and critical-path exposed
time.

## Out of scope

- a separate long-form per-edge CSV;
- per-event trace export;
- new physical GPU-to-HBM or HBM-to-HBM PHY resources;
- critical-path attribution;
- changing independent-path timing semantics;
- new topology, routing, placement, capacity, thermal, PDN, or energy models;
- bank/channel/address-level HBM simulation.
