# Layer-Prefetch Memory Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible eight-lane B0/B2 layer-prefetch simulator that measures whether chunked weight/KV movement is hidden by full-layer execution.

**Architecture:** Preserve the existing Stage 2 engine as the default legacy path. Build a focused prefetch workload/config/resource/simulator stack that reuses the existing Llama-2 footprint math, D-2 semantic compute/collective schedule, D-1 collective resources, and a common tagged `RateResource`; expose aggregate sweep rows plus optional detailed traces.

**Tech Stack:** Python 3, SimPy, dataclasses, PyYAML, pytest, CSV

**Spec:** `docs/superpowers/specs/2026-09-30-layer-prefetch-memory-hierarchy-design.md`

## Global Constraints

- Existing YAML files, legacy `run_simulation()`, D-1/D-2/D-3 behavior, CSV schemas, and feasible-point numerical results remain unchanged.
- Legacy regression reference is corrected baseline commit `b88a284` (`219 passed`, sanity `8/8 PASS` before feature work); comparisons must use that commit rather than the original repository HEAD or the dirty parent checkout.
- Nominal topology is exactly eight symmetric lanes per GPU: B0 has 8x48 GB conventional HBM; B2 has 8x12 GB uHBM plus 8x48 GB side HBM.
- Prefetch/side links never consume D-1 inter-GPU collective H2H resources.
- Full-layer-ready is mandatory; no intra-layer streaming.
- All memory/link requests are non-preemptive FCFS at configured chunk granularity; no priority or fixed directional bandwidth partition.
- Capacity/link numbers are direct config parameters, not PDN/SI/thermal/distance-derived values.
- No production code is written before its owning test has been observed failing for the intended reason.

## Review Focus

- A transfer total not divisible by eight lanes or the chunk size must conserve the original bytes through residual-corrected final lane/chunk parts; Task 2 and Task 4 test this.
- A B2 KV writeback that has left uHBM but not reached side HBM must still block the next-token read of that layer; Task 5 tests this causal boundary.
- `persistent_if_fit` must reserve mandatory runtime state before deciding persistence and must atomically fall back to streaming rather than partially pinning weights; Task 3 tests this.
- `steady_state` must exclude warm-up timing without fabricating a pre-ready layer-0 buffer or using the relaxed collective DAG; Task 5 tests this.
- Tagged resource accounting must preserve legacy total accumulation order and produce no unclassified production requests; Task 1 and Task 7 test this.

---

## File structure

- Modify `uhbm_simulator/resources.py`: common traffic tags and per-tag resource accounting; existing total stats remain authoritative.
- Modify `uhbm_simulator/simulator.py`: immutable tagged metric snapshots only; legacy scheduling stays unchanged.
- Create `uhbm_simulator/prefetch_config.py`: dedicated YAML schema and validation for the new hierarchy.
- Create `uhbm_simulator/prefetch_workload.py`: per-layer/lane footprints, KV growth, weight-residency decisions, and residual splits.
- Create `uhbm_simulator/prefetch_resources.py`: B0/B2 eight-lane resources and chunk pipeline primitives.
- Create `uhbm_simulator/prefetch_simulator.py`: full-layer-ready prefill/decode event graph and result/trace types.
- Create `uhbm_simulator/prefetch_sweep.py`: Cartesian expansion, summary rows, energy, CSV writer, and module CLI.
- Create `configs/stage2_layer_prefetch_example.yaml`: runnable, explicitly scenario-valued schema example, not a calibrated production profile.
- Modify `README_stage2.md`: architecture, assumptions, execution, metrics, and calibration caveats.

### Task 1: Tagged resource-accounting foundation

**Files:**
- Modify: `uhbm_simulator/resources.py`
- Modify: `uhbm_simulator/simulator.py`
- Create: `tests/test_path_accounting.py`

**Interfaces:**
- Produces: `TrafficClass` string constants, `RateResource.use(amount, traffic_class="unclassified")`, `ResourceMetric.by_traffic_class`.
- Preserves: every existing top-level `ResourceStats` and `ResourceMetric` value.

- [ ] **Step 1: Write failing compatibility and attribution tests**

Add tests named `test_untagged_use_preserves_legacy_totals`, `test_tagged_requests_attribute_bytes_and_wait_time_to_the_request_class`, and `test_tagged_subtotals_conserve_resource_totals`. Assert the same request updates legacy totals once, queue time belongs to the request that waited, and `math.fsum` of tag bytes/service/latency equals the total.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_path_accounting.py -q`

Expected: FAIL because `traffic_class` and `by_traffic_class` do not exist.

- [ ] **Step 3: Implement tagged accounting without recomputing legacy totals**

Add declared traffic-class validation, per-class `ResourceStats`, and immutable snapshots. Keep the existing total update statements and their order unchanged.

- [ ] **Step 4: Run focused and legacy resource tests**

Run: `python -m pytest tests/test_path_accounting.py tests/test_resources.py tests/test_simulator_materialized.py tests/test_simulator_cut_through.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the foundation**

Commit: `feat: add tagged resource accounting`

### Task 2: Prefetch configuration and workload decomposition

**Files:**
- Create: `uhbm_simulator/prefetch_config.py`
- Create: `uhbm_simulator/prefetch_workload.py`
- Create: `tests/test_prefetch_config.py`
- Create: `tests/test_prefetch_workload.py`
- Create: `configs/stage2_layer_prefetch_example.yaml`

**Interfaces:**
- Produces: `PrefetchSimulationConfig`, `ArchitectureConfig`, `CapacityMemoryConfig`, `PrefetchPolicyConfig`, `PrefetchWorkloadConfig`, `load_prefetch_config(path)`.
- Produces: `LayerFootprint`, `LaneLayerPlan`, `WeightResidencyDecision`, `build_layer_plans(...)`, `split_residual(total, count)`.
- Consumes: `stage1_model.MODELS`, `b_weight_footprint`, `b_kv_footprint`, and D-2 attention/MLP parameter attribution.

- [ ] **Step 1: Write failing schema tests**

Test exact parsing of eight lanes, B0 48e9 bytes/lane, B2 uHBM 12e9 and side HBM 48e9 bytes/lane, three independent chunk sizes, nullable in-flight depth, `decode_tokens`, run/start modes, and technology-neutral capacity-memory/link fields. Test rejection of lane counts other than eight, missing B2 side fields, nonpositive chunk sizes, invalid modes, and capacity too small for the striped weight backing copy.

- [ ] **Step 2: Verify config tests RED**

Run: `python -m pytest tests/test_prefetch_config.py -q`

Expected: FAIL because the prefetch config module does not exist.

- [ ] **Step 3: Implement the dedicated loader and example config**

Keep all numeric BW/latency/energy values explicit in YAML. Label the example values as scenario inputs; do not name them measured DDR/CXL/uHBM presets.

- [ ] **Step 4: Write failing workload-conservation tests**

Test that per-layer totals conserve model weight/KV, every lane gets exactly one eighth under uniform mapping including a residual final lane, fresh prefill has zero historical-KV prefetch, decode KV grows by one token, and `decode_tokens` must be positive.

- [ ] **Step 5: Verify workload tests RED, then implement the minimal decomposition**

Run RED then GREEN: `python -m pytest tests/test_prefetch_workload.py -q`

- [ ] **Step 6: Run both focused suites**

Run: `python -m pytest tests/test_prefetch_config.py tests/test_prefetch_workload.py -q`

Expected: PASS.

- [ ] **Step 7: Commit config and workload planning**

Commit: `feat: define layer prefetch workloads and config`

### Task 3: Capacity and residency policy

**Files:**
- Modify: `uhbm_simulator/prefetch_workload.py`
- Create: `tests/test_prefetch_capacity.py`

**Interfaces:**
- Produces: `decide_weight_residency(architecture, workload, policy) -> WeightResidencyDecision` and typed infeasibility reasons.
- Consumes: Task 2 footprint and config types.

- [ ] **Step 1: Write failing B0/B2 capacity tests**

Cover B2 mandatory current/next KV plus scratch before weight persistence; B0 full historical KV before weight persistence; exact-fit persistence; one-byte-short atomic streaming fallback; streaming double-buffer infeasibility; side-HBM KV overflow; and decoded-token capacity growth reporting the first infeasible token.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_prefetch_capacity.py -q`

Expected: FAIL because residency decisions are not implemented.

- [ ] **Step 3: Implement mandatory-state-first decisions**

Do not reuse legacy `r_remote` or `capacity_overflow`; the new hierarchy has explicit tier roles. Do not implement partial weight pinning.

- [ ] **Step 4: Verify focused capacity and workload suites**

Run: `python -m pytest tests/test_prefetch_capacity.py tests/test_prefetch_workload.py -q`

Expected: PASS.

- [ ] **Step 5: Commit residency policy**

Commit: `feat: add prefetch hierarchy capacity policy`

### Task 4: Eight-lane resources and chunk pipeline

**Files:**
- Create: `uhbm_simulator/prefetch_resources.py`
- Create: `tests/test_prefetch_resources.py`
- Create: `tests/test_chunk_pipeline.py`

**Interfaces:**
- Produces: `PrefetchResourceLayer`, `build_prefetch_resources(config, env)`, `transfer_chunks(...)`.
- Consumes: tagged `RateResource`, Task 2 configs, `split_residual`.

- [ ] **Step 1: Write failing topology tests**

Assert B0 creates eight capacity memories, eight capacity links, and eight conventional HBM resources with no side resources. Assert B2 creates eight capacity memories/links, eight uHBMs, eight side HBMs, and eight shared bidirectional side links. Assert no prefetch resource aliases D-1 H2H resources.

- [ ] **Step 2: Verify topology RED, then implement resource construction**

Run RED then GREEN: `python -m pytest tests/test_prefetch_resources.py -q`

- [ ] **Step 3: Write failing pipeline tests**

Use unit-rate toy resources to assert residual byte conservation, source/link/destination pipeline overlap for depth greater than one, no inter-chunk pipeline at depth one, unbounded enqueue behavior, fixed latency application per chunk, and FCFS ordering when KV prefetch and writeback share a side link.

- [ ] **Step 4: Verify pipeline RED, then implement `transfer_chunks`**

Run RED then GREEN: `python -m pytest tests/test_chunk_pipeline.py -q`

- [ ] **Step 5: Verify resource/pipeline suites together**

Run: `python -m pytest tests/test_prefetch_resources.py tests/test_chunk_pipeline.py tests/test_resources.py -q`

Expected: PASS.

- [ ] **Step 6: Commit resource engine**

Commit: `feat: add eight-lane chunked memory resources`

### Task 5: Full-layer-ready simulator and token state

**Files:**
- Create: `uhbm_simulator/prefetch_simulator.py`
- Create: `tests/test_layer_prefetch_simulator.py`
- Create: `tests/test_prefetch_token_state.py`

**Interfaces:**
- Produces: `PrefetchSimulationResult`, `LayerTiming`, `TokenTiming`, `PrefetchTrace`, `run_prefetch_simulation(config, *, detailed_trace=False)`.
- Consumes: Tasks 2-4 plans, residency decisions, resources, and the existing strict D-2/D-1 compute/collective semantics.

- [ ] **Step 1: Write failing full-layer-ready toy tests**

Assert next-layer prefetch launches at current-layer start, compute waits for all weight/KV chunks across all eight lanes, the slowest lane determines readiness, positive slack creates no stall, negative slack adds exactly its magnitude, and no compute consumes partial layer data.

- [ ] **Step 2: Verify RED, then implement one-token B2 scheduling**

Run RED then GREEN: `python -m pytest tests/test_layer_prefetch_simulator.py -q`

- [ ] **Step 3: Add failing B0/B2 path-semantic tests**

Assert B0 has capacity-to-2.5D weight prefetch and no KV-prefetch transport; B2 has capacity-to-uHBM weight prefetch, side-to-uHBM KV prefetch, and uHBM-to-side writeback; both retain strict collective dependency.

- [ ] **Step 4: Implement both architecture paths and verify GREEN**

Run: `python -m pytest tests/test_layer_prefetch_simulator.py -q`

- [ ] **Step 5: Write failing multi-token and phase-state tests**

Cover fresh prefill KV generation, `decode_only` initial resident KV, prefill-to-decode state transfer, new-KV writeback blocking the next-token read until side-HBM completion, increasing KV resident bytes, `cold_start` layer-0 preparation, and `steady_state` warm-up exclusion with strict causality.

- [ ] **Step 6: Implement token state and verify GREEN**

Run: `python -m pytest tests/test_prefetch_token_state.py tests/test_layer_prefetch_simulator.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the event simulator**

Commit: `feat: simulate full-layer-ready prefetch execution`

### Task 6: Metrics, path energy, sweep, and CLI

**Files:**
- Create: `uhbm_simulator/prefetch_sweep.py`
- Create: `tests/test_prefetch_metrics.py`
- Create: `tests/test_prefetch_sweep.py`

**Interfaces:**
- Produces: `PrefetchSweepRow`, `run_prefetch_sweep(config)`, `write_prefetch_csv(rows, path)`, and `python -m uhbm_simulator.prefetch_sweep --config ... --output ...`.
- Consumes: `PrefetchSimulationResult` and tagged resource metrics.

- [ ] **Step 1: Write failing path-metric and energy tests**

Assert per-path bytes/request/queue/service/latency values, local-memory access-class separation, critical-lane selection, mean/max utilization, actual-byte energy, side-link bidirectional accounting, no crossing-energy double count, and total dynamic-energy conservation.

- [ ] **Step 2: Verify RED, implement metrics, verify GREEN**

Run: `python -m pytest tests/test_prefetch_metrics.py -q`

- [ ] **Step 3: Write failing sweep/CSV tests**

Assert Cartesian expansion of architecture, batch, decode tokens, start state, weight policy, chunk sizes, in-flight depth, and capacity-memory profile; verify infeasible rows retain identifying fields and use `None` for simulation metrics; verify no token/layer/lane detail explodes the summary CSV.

- [ ] **Step 4: Implement sweep and CLI, then verify GREEN**

Run: `python -m pytest tests/test_prefetch_sweep.py -q`

- [ ] **Step 5: Run a toy end-to-end CLI point**

Run: `python -m uhbm_simulator.prefetch_sweep --config configs/stage2_layer_prefetch_example.yaml --output results/stage2_layer_prefetch_example.csv`

Expected: command completes, emits both B0 and B2 rows, and labels scenario parameters as uncalibrated.

- [ ] **Step 6: Commit reporting and CLI**

Commit: `feat: report layer prefetch sweeps`

### Task 7: Documentation and full regression verification

**Files:**
- Modify: `README_stage2.md`
- Modify: `docs/superpowers/specs/2026-09-30-path-segment-accounting-design.md`
- Test: all existing and new tests

**Interfaces:**
- Documents: the new command, B0/B2 topology, capacities, policies, cold/steady semantics, path queries, CSV fields, and all Stage 2 caveats.

- [ ] **Step 1: Document the new engine and supersession relationship**

State that the legacy engine remains default; explain full-layer-ready, fresh prefill, continuous decode, warm-model persistence, chunk/in-flight bounds, technology-neutral profiles, and resource-time versus critical-path time.

- [ ] **Step 2: Add the production-classification regression**

Add `test_prefetch_production_paths_have_no_unclassified_transfer_requests` and ensure every new request site carries a declared tag.

- [ ] **Step 3: Run all new feature tests**

Run: `python -m pytest tests/test_path_accounting.py tests/test_prefetch_config.py tests/test_prefetch_workload.py tests/test_prefetch_capacity.py tests/test_prefetch_resources.py tests/test_chunk_pipeline.py tests/test_layer_prefetch_simulator.py tests/test_prefetch_token_state.py tests/test_prefetch_metrics.py tests/test_prefetch_sweep.py -q`

Expected: PASS.

- [ ] **Step 4: Run the full existing test suite**

Run: `python -m pytest -q`

Expected: all tests PASS with no warnings attributable to the change.

- [ ] **Step 5: Run legacy sanity**

Run: `python -m uhbm_simulator --sanity`

Expected: existing 8/8 PASS and unchanged detail values.

- [ ] **Step 6: Compare representative legacy outputs bit-for-bit**

Run the existing default config to a temporary output and compare all pre-existing columns by keyed rows against corrected baseline commit `b88a284`. Do not use the repository's original initial commit, and do not overwrite committed/user result CSVs.

- [ ] **Step 7: Commit documentation and verification locks**

Commit: `docs: document layer prefetch hierarchy`

## Self-review record

- Spec coverage: topology, placement, capacity, residency, phases, cold/steady, chunking, FCFS resources, capacity profiles, metrics, energy, compatibility, and scope exclusions all map to Tasks 1-7.
- Type consistency: config -> workload -> resources -> simulator -> sweep interfaces are defined once in dependency order.
- Review-focus coverage: every listed failure mode has an explicit named test in its owning task.
- Proportion: algorithms are specified by invariants and signatures; implementation bodies remain with the executor.
