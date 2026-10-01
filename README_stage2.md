# uHBM Stage 2 discrete-event simulator

## Capacity-feasibility sweep

`configs/stage2_breakeven_map.yaml` keeps the default Llama-70B workload and
expands `local_hbm.count` to `[1, 2, 4]`. Its purpose is to retain capacity-
infeasible design points in the break-even map instead of avoiding them with a
smaller model.

The CSV column `capacity_feasible` reports whether every local-HBM stack can
hold its assigned working set and materialized receive buffer. When it is
`False`, simulation-derived fields such as TTFT, TPOT, communication/overlap,
utilization, and queueing time are empty; design metadata and the independent
H2H analytical-feasibility fields remain available. `architecture_feasible`
continues to mean H2H bandwidth/latency feasibility only.

**Scope lock:** `uniform` and `weighted` remain declarative per-stack traffic
distribution policies. They do not implement bit-level address mapping or
interleaving, address hashing, or HBM bank/channel conflicts; those remain
Stage 4 work.

`uhbm_simulator/`는 `uhbm_simulator_spec_v3.md`의 Stage 2를 구현한 순수 Python/SimPy
시뮬레이터다. `stage1_model.py`의 `b_local_per_token`과 `b_remote_per_token`을 직접
재사용한다. BookSim, Accel-Sim, ASTRA-Sim은 import, 빌드, 실행하지 않는다.

## Model preset scope

`"Llama-70B"`와 `"Llama-7B"` preset 이름은 하위 호환을 위해 유지하지만, 이
시뮬레이터에서 두 preset은 각각 **Llama-2-70B**와 **Llama-2-7B** 아키텍처를 뜻한다.
Hidden size, layer/head 수와 70B의 GQA group size는 Touvron et al., *Llama 2:
Open Foundation and Fine-Tuned Chat Models* (2023)의 Table 1/Section 2.2 및 Meta의
공식 Llama-2 config를 따른다. Llama 1이나 Llama 3는 head/GQA 구성이 다르므로 이
preset으로 대표하지 않는다.

Llama-2-70B는 64 attention heads와 8 KV heads를 사용하므로 KV 차원은 MHA 식의
정확히 1/8이다. Llama-2-7B는 32 attention heads와 32 KV heads를 사용하므로 기존
MHA 식과 동일하다. 이 1/8 비율은 **KV 항에만** 적용된다. Weight 항이 섞인 전체
`b_local`, capacity 및 에너지 파생값은 workload별 weight/KV 구성에 따라 다른 비율로
변한다. 또한 dense-attention decode가 과거 KV 전체를 읽는다는 가정은 유지된다.
Paged KV, eviction, sliding-window/sparse attention은 모델링하지 않는다. Prefill도 같은
GQA-aware helper를 재사용하지만, 이는 기존의 초기 prefill local-memory 근사를 검증하거나
정교화했다는 뜻은 아니다.

## 실행

Windows PowerShell 또는 WSL/Linux에서 Python 3.10 이상을 사용한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uhbm_simulator --sanity
.\.venv\Scripts\python.exe -m uhbm_simulator `
  --config configs\stage2_default.yaml `
  --output results\stage2_materialized.csv
```

WSL/Linux에서는 실행 파일을 `.venv/bin/python`으로 바꾸면 된다.

`comm_mode`를 `cut_through`로 바꿔 upper-bound 경로를 실행할 수 있다. 기본값과 검증
기준은 `materialized`다.

## Dedicated layer-prefetch memory hierarchy

The legacy Stage 2 engine and its existing YAML/CSV schema remain the default.
The opt-in `layer_prefetch` engine answers a narrower question: whether all
weight and historical-KV chunks for the next transformer layer can reach an
eight-lane local-memory staging tier before that layer may execute. It uses a
full-layer-ready barrier; intra-layer streaming into compute is deliberately
not modeled.

The dedicated B0 and B2 abstractions are symmetric and lane-paired:

- B0 has eight 48 GB conventional 2.5D-HBM lanes. Capacity memory feeds each
  paired HBM directly; KV remains resident in that HBM.
- B2 has eight 12 GB uHBM lanes, eight paired 48 GB side-HBM devices, and eight
  paired capacity-memory devices. Weight arrives directly from capacity memory
  to uHBM. Historical KV is prefetched from paired side HBM to uHBM and newly
  generated KV is written back over the same bidirectional, FCFS side link.

Capacity memory is a per-lane backing store. Its device bandwidth/latency/
energy and the longer capacity-to-local link are explicit scenario inputs.
They are technology-neutral placeholders, not calibrated DDR, CXL, HBM, or
uHBM product specifications. The example profile is therefore named
`uncalibrated_example`.

`persistent_if_fit` first attempts atomic full-model weight persistence after
mandatory KV/staging capacity is reserved; if it does not fit, the model falls
back to layer streaming. `streaming` always fetches layer weight. Transfers are
split into the configured weight/KV/writeback chunks. A finite
`max_inflight_chunks_per_lane` shares descriptors between prefetch and
writeback; `null` leaves descriptor count unbounded while the byte-rate
resources still serialize traffic.

`fresh prefill` starts without fabricated prefetched data. Decode is continuous:
KV grows token by token and B2 writes new KV through uHBM to side HBM. A
`cold_start` includes layer-0 fetch cost. `steady_state` executes one causally
valid strict-D1/D2 warm-up token and excludes only that token from reported
decode timing; it does not use the relaxed collective DAG or assume a free
layer-0 buffer. `pp=1` remains fixed.

Run the small two-point B0/B2 example with:

```powershell
.\.venv\Scripts\python.exe -m uhbm_simulator.prefetch_sweep `
  --config configs\stage2_layer_prefetch_example.yaml `
  --output results\stage2_layer_prefetch_example.csv
```

The sweep is a true Cartesian product over architecture, capacity-memory
profile, phase/start state, batch, decode-token count, weight policy, the three
chunk sizes, and in-flight depth. Infeasible points remain in the CSV with
identity and capacity reason populated; simulation-derived values are empty.
Summary rows contain latency/throughput, residency, dynamic energy, local-lane
mean/max utilization and the critical lane. They also report decode-token
mean/median/maximum latency, cold/steady timing context, total/weight/KV
prefetch stall, hidden-prefetch fraction, deadline-miss count, the limiting
prefetch class, and KV-writeback deadline lateness. Weight and KV stall totals
are independent causal diagnostics and can overlap; they must not be added to
attribute critical-path time. `path_metrics_json` contains each
declared path's actual bytes, request count, accumulated queue/service/fixed-
latency time, and dynamic energy. Token/layer/lane traces are available only
through `run_prefetch_simulation(..., detailed_trace=True)` and are not copied
into every CSV row.

Example path query:

```python
from uhbm_simulator.prefetch_sweep import calculate_prefetch_metrics
from uhbm_simulator.resources import TrafficClass

result = run_prefetch_simulation(point, detailed_trace=True)
paths = calculate_prefetch_metrics(point, result).paths
gpu_reads = paths[TrafficClass.UHBM_GPU_READ]
side_prefetch = paths[TrafficClass.SIDE_TO_UHBM]
collective = paths[TrafficClass.COLLECTIVE_H2H_TRANSPORT]
```

These times are sums charged to resources, not exposed critical-path time.
They may exceed `elapsed_s` because lanes and paths overlap. The simulator does
not infer energy from utilization: every dynamic term is actual transferred
bytes times the configured pJ/bit for that physical device or link segment.
Side-HBM datapath and side-link crossing are separate segments, so neither is
silently omitted or charged twice. H2H and GPU-direct collective transport use
the dedicated execution profile's nominal energy anchors; all such scenario
numbers remain placeholders until Stage 3/4 calibration.

The model still does not simulate bit-level address mapping, HBM banks/channels,
page migration, predictors for dynamic expert selection, PDN/IR-drop, thermal
physics, EM, or packet routers. Layer order is statically known for the dense
Llama-2 presets; the engine evaluates transfer feasibility and scheduling, not
whether an arbitrary future layer can be predicted.

## Sensitivity analysis

다음 명령은 한 번에 한 축만 바꾸는 OAT(one-at-a-time) 분석을 실행하고 축당 한 행을
`results/stage2_sensitivity.csv`에 기록한다.

```powershell
.\.venv\Scripts\python.exe -m uhbm_simulator.sensitivity `
  --config configs\stage2_default.yaml `
  --output results\stage2_sensitivity.csv
```

기준점은 `r=0.5`, `r_remote=0.1`, `comm_mode=materialized`,
`thermal_scenario=no_throttling`, `local_hbm_count=4`, `uniform`, 24 Gbps/lane,
`n_pillar=8`, `batch=8`이며 기본 Llama-70B 설정에서 capacity-feasible하다. batch 중앙값이
capacity를 초과하면 중앙에서 가까운 순서로 첫 feasible 값을 선택한다. `r`, `r_remote`, H2H
lane rate, `n_pillar`는 입력 config의 리스트를 사용한다. `local_hbm_count`는 config의
`1..max_count`, categorical 축은 검증된 전체 domain(materialized/cut-through,
thermal 3개 scenario, uniform/weighted)을 사용한다. 각 축 이외의 값은 모두 기준점에
고정한다. `batch`는 config에 정의된 전체 리스트를 사용한다. 여기에 전체 Cartesian
sweep에는 넣지 않는 에너지 전용 OAT 축 두 개가 추가된다.

- `side_hbm.link_energy_pj_per_bit`: `[0.54, 10.0, 20.0, 40.0]`, baseline `10.0`
- `local_hbm.stack.hbm_idle_power_w`: `[0.4, 2.0, 5.2]`, baseline `0.4`
- `side_hbm_path`: values from the input config; the production config is the
  singleton `shared_h2h`, while `configs/stage2_side_hbm_path_sensitivity.yaml`
  compares `[shared_h2h, independent]` explicitly.

Because the OAT baseline is `shared_h2h`, changing only
`side_hbm.link_energy_pj_per_bit` has zero energy effect: that separate link
term is disabled and the crossing is already charged through H2H edge bytes.
The 0.54/10/20/40 pJ/bit values affect only an `independent` side path; this is
a conditional model semantic, not evidence that crossing energy is irrelevant.

따라서 현재 sensitivity CSV는 총 12개 축을 기록한다. 두 baseline은
`stage2_default.yaml`의 스칼라 nominal 값과 동일하며 메인 전체 sweep의 조합 수를
늘리지 않는다.

`tpot_s_relative_range`와 `communication_time_s_relative_range`는 capacity-feasible
포인트만 사용한 `(max-min)/baseline`이다. `capacity_infeasible_fraction`과
`n_capacity_infeasible_points`는 값이 없는 포인트의 비율과 개수다.

같은 방식으로 `energy_per_token_j_min/max/relative_range`와
`decode_throughput_tokens_per_s_min/max/relative_range`도 축별로 기록한다. 두 relative
range 역시 capacity-feasible 포인트만 사용한 `(max-min)/baseline`이며 별도 baseline
해석이나 architecture shadow evaluation을 적용하지 않는다. Prefill workload에서는
`decode_throughput_tokens_per_s`가 모든 포인트에서 `None`이므로 해당 min/max/range도
모두 빈 값으로 기록된다.

`flips_architecture_feasible`은 항상 `h2h_feasibility()`의 **native `r=0` semantics**로
shadow 평가하며, OAT row의 실제 `r` 값과 무관하다. 따라서 `r` 축 자체에서는 이 값이
구조적으로 상수이고 flip이 없는 것이 예상 동작이지 버그가 아니다.
`r_remote`, `comm_mode`, `thermal_scenario`, `address_mapping.policy` 및 위 두 에너지
전용 축도
`h2h_feasibility()`의 입력이 아니므로 이 지표의 flip은 구조상 항상 `False`다. 이를
"실험 결과 해당 축이 architecture에 영향 없음"으로 해석하면 안 된다. CSV의
`architecture_feasibility_applicable=False`가 이러한 비적용을 표시하고,
`local_hbm_count`, `h2h.lane_rate_gbps`, `h2h.n_pillar`, `batch`만 `True`다. 이 중
`batch`는 workload traffic signature를 바꾸고 그 signature가 `h2h_feasibility()`에
입력되므로 적용 대상으로 분류된다.

`address_mapping.policy=weighted` 포인트는 `[0.7, 0.1, 0.1, 0.1]`을 사용한다. 이는
평균적인 weighted 분포가 아니라 **의도적으로 만든 worst-case/hotspot profile**이다.
기본 Llama-70B 결과에서는 이 weighted 포인트가 capacity-infeasible이므로 효과가 metric
elasticity가 아니라 `flips_capacity_feasible=True`로 나타난다. 향후 feasible한 weighted
포인트에서 큰 elasticity가 나오더라도 일반적인 weighted mapping의 효과가 아니라 hotspot
민감도로 해석해야 한다. 이 설명은 CSV의 `notes`에도 함께 기록된다. 실제 bit-level 주소
매핑이나 bank/channel conflict를 구현한다는 뜻은 아니다.

## 트래픽 분리

기본 `fixed_fraction` 모드에서 두 입력 비율은 서로 독립적이다.

```text
b_local  -> (1-r_remote) local_hbm + r_remote side_hbm
b_remote -> (1-r) h2h_link + r gpu_direct_link
```

`fixed_fraction`에서 `r_remote`는 access fraction이 아니라, `b_local` 중 side-HBM에
**배치되어 상주하는 비율(placement fraction)**이다. 접근은 그 배치를 그대로 따른다고
가정하며 side-HBM은 capacity-unconstrained pooled/remote memory로 취급한다.
`capacity_overflow`에서는 이 입력을 무시하고 class별 resident footprint와 local capacity로
분할을 유도한다. 이 모드에서만 `side_hbm_capacity_bytes`를 선택적으로 지정해 side resident
footprint를 제한할 수 있으며, `None`이면 side-HBM 용량은 제한하지 않는다.

## GPU별 local-HBM stack

`local_hbm`은 GPU당 aggregate 자원 하나가 아니라 `(tile, stack_index)`로 구분된 개별
HBM stack 자원이다. 기본 구성은 GPU 아래 2x2로 배치된 HBM 4개이며, 각 HBM은 내부적으로
4-high stack이다. 기본 per-stack 대역폭/용량은 1.2 TB/s와 16 GB이므로 uniform 4-stack의
총 대역폭과 용량은 기존 모델과 동일한 4.8 TB/s와 64 GB이다.

`SimulationParameters.local_hbm_latency_s`는 각 local-HBM stack 접근에 적용되는 고정
latency다. GPU에 물리적으로 매우 가까운 uHBM local stack은 지금까지와 같이 이를 `0.0`으로
근사하며, 이 값이 기본값이므로 기존 config와 결과의 동작은 바뀌지 않는다. 이 파라미터는
향후 conventional 2.5D HBM처럼 유의미한 local-memory access latency가 있는 B0 baseline을
동일한 시뮬레이터에서 B1/B2 uHBM과 비교할 수 있도록 마련한 것이다. B0/B1/B2의 실제 수치와
config 정의는 이번 변경 범위에 포함되지 않는다.

```yaml
local_hbm:
  stack:
    bw_bps: 1.2e12
    capacity_bytes: 16e9
  count: [4]
  max_count: 4
  address_mapping:
    policy: uniform
    weights: null
```

`count`는 sweep 축이며 물리적 2x2 상한 때문에 1~4만 허용한다. 기본 config는 기존 240개
설계점을 유지하도록 `[4]`만 사용하고, `configs/stage2_hbm_count_sweep.yaml`은 `[1, 2, 4]`
예시를 제공한다. 이 예시는 1-stack의 16 GB 용량에서도 모든 설계점이 실행되도록
Llama-7B를 사용한다.

`uniform`은 각 stack에 같은 바이트 수를 보내고, `weighted`는 단일 count에 대해 설정한
비음수 weight를 정규화하여 의도적인 hotspot을 만든다. CSV의
`local_hbm_utilization`은 모든 stack의 평균이고 `local_hbm_utilization_max`는 가장 바쁜
stack의 사용률이다.

`local_hbm_placement_mode`의 production 기본값은 `fixed_fraction`이다. 이 모드에서는
입력 `r_remote`가 기존과 동일하게 local/side resident footprint와 각 memory traffic을
고정 비율로 나누므로 기존 config와 수치가 그대로 유지된다. `capacity_overflow`를 명시하면
입력 `r_remote`는 완전히 무시되며, local stack 용량 합계에 실제로 들어가는 class별 resident
footprint로 placement를 결정한다. 결과 traffic 기준 remote 비율은
`TrafficStreams.effective_r_remote`로 제공된다.

`capacity_overflow`의 배치 순서는 weight -> KV -> activation이다. 이 순서는 NUMA-Aware
GPUs 등의 문헌이 제시한 최적 정책이 아니라, capacity-driven placement를 재현 가능하게
평가하기 위한 deterministic MVP modeling choice다. 향후 priority 자체를 설정 또는 sweep하는
기능은 별도 작업이며 현재 구현에는 LRU, paging, eviction이 없다. 각 class의 local placement
fraction은 반드시 그 class 자신의 traffic에만 적용하며 resident footprint와 transferred
traffic을 합치지 않는다.

Capacity-overflow placement는 mapping policy와 일치하는 usable local capacity를 사용한다.
Uniform은 aggregate physical capacity를 사용하고, weighted는 가장 큰 stack fraction이
허용하는 capacity를 사용한다. 기존 per-stack 검사는 계산 오류와 buffer 자체의 초과를 잡는
downstream safety check로 유지한다. Optional `side_hbm_capacity_bytes`도 이 모드에서만 side
resident footprint를 제한하며, `None`이면 기존처럼 side HBM을 capacity-unconstrained
pooled/remote memory로 취급한다.

Materialized communication에서는 placement 전에 receive-buffer headroom
`H_materialized = max_stage(B_H2H,collective / (2*(TP-1)))`를 예약한다. 이는
sequential ring round 사이에 재사용되는 per-GPU round scratch buffer의 peak-live
크기다. Uniform mapping의 usable capacity는
`N * C_stack - H_materialized`이고, weighted mapping은 가장 바쁜 stack의 downstream
constraint와 일치하도록 `C_stack / max(stack_fraction) - H_materialized`를 사용한다.
두 값 모두 0 아래로 내려가면 0으로 clamp하며, buffer 자체가 stack capacity를 넘는 경우에는
기존 post-placement per-stack safety check가 `LocalHBMCapacityError`를 발생시킨다.
Cut-through에는 materialized receive buffer가 없으므로 이 예약으로 값이 바뀌지 않는다.

### Attention/MLP semantic stages

Llama workload는 각 layer를 attention과 MLP의 두 semantic stage로 나눈다. Stage
`2 * layer_index`는 attention이고 stage `2 * layer_index + 1`은 MLP이므로,
Llama-2-70B는 80 layer에 대해 160개 stage를 갖는다. Weight traffic은 공식
Llama-2 config의 GQA-aware attention projection parameter 수와 SwiGLU MLP
parameter 수로 유도한 비율에 따라 두 stage 종류로 나뉜다. Preset의
`intermediate_size`는 Llama-2-7B 11008, Llama-2-70B 28672이다. 이 분해는
parameter-count 기반 first-order attribution이며 detailed kernel model은 아니다.
전체 `n_params`에 포함된 embedding, normalization, output-head weight도 block에서
유도한 같은 비율로 귀속시켜 aggregate weight traffic을 보존한다.

KV resident footprint는 기존처럼 capacity placement class로 유지하지만, KV traffic은
attention stage에만 배정한다. Activation traffic은 attention/MLP 상세 분해 근거가
없으므로 aggregate traffic을 `2 * n_layers` stage에 균등 배분한다. Footprint는 stage로
나누지 않으며 capacity accounting에만 사용한다. 따라서 resident footprint와 transferred
traffic은 계속 서로 다른 물리량이다.

`n_side_requests`는 semantic Llama workload의 입력 scheduling parameter가 아니다.
Placement 후 실제 side traffic이 0보다 큰 stage의 수로 파생된다. 예를 들어 KV만
remote이면 attention stage에만 요청이 생겨 `n_layers`개이고, weight가 일부라도
remote이면 attention과 MLP 양쪽에 weight traffic이 있으므로 `2 * n_layers`개다.
`semantic_stages=None`인 legacy/custom `TrafficSignature`만 기존 aggregate request-count
hint와 scheduler를 그대로 사용한다.

현재 프로젝트의 Megatron-style aggregate collective abstraction에서는 attention 뒤와
MLP 뒤 logical AllReduce가 모두 같은 hidden-state payload를 사용한다고 가정한다. 따라서
각 stage의 payload를 `B_remote / (2L)`로 두는 것은 **이 abstraction 내부에서는** total
collective bytes를 정확히 보존한다. 이것은 모든 실제 구현에서 두 payload가 같다는 일반적
주장이 아니다. Sequence parallelism, operator/communication fusion 등 구현 세부사항에
따라 실제 payload는 달라질 수 있으며, 향후 상세 모델 보정 대상이다. D-2 stage는 logical
collective 경계만 나타내며 ring round는 후속 작업에서 각 logical collective 아래에 붙인다.

### Local/side memory concurrency bounding cases

`SimulationParameters.local_side_memory_concurrency`는 같은 workload stage의 local-HBM과
side-HBM 요청 사이 concurrency를 두 개의 event-driven 경계값으로 표현한다. 기본값
`full_overlap`은 기존 동작을 bit-exact하게 유지하며 두 요청을 동시에 시작해 모두 끝날 때까지
기다린다. 이는 성능 관점의 optimistic bound이자 latency lower bound로,
`T_memory ≈ max(T_local, T_side)`에 해당한다. `serialized`는 local 요청이 끝난 뒤 side 요청을
시작하는 conservative no-concurrency bound이자 latency upper bound로,
`T_memory = T_local + T_side`에 해당한다. Local-first 순서는 물리적 arbitration 정책을
주장하는 것이 아니라 deterministic bounding schedule이다.

실제 uHBM의 동작은 다음 범위 안에 있을 가능성이 높지만, 어느 지점인지는 아직 검증되지 않았다.

```text
max(T_local, T_side) <= T_actual <= T_local + T_side
```

`serialized`에서도 다음 stage의 local→side memory chain은 이전 collective와 overlap할 수 있고,
strict dependency는 둘이 모두 완료된 뒤 compute를 시작한다. 이 모드는 같은 stage의 workload
local/side 요청만 제어한다. Materialized communication endpoint의 local-HBM 접근까지 포함하는
공유 fabric, tile 간 전역 직렬화 또는 partial-overlap 계수는 모델링하지 않는다. 두 모드 중
어느 것도 실측으로 검증된 현실적 기본값으로 해석하면 안 된다.

**Scope lock:** `address_mapping.policy`라는 이름은 실제 주소 매핑 구현을 뜻하지 않는다.
이 기능은 트래픽 분배 결과를 declarative policy/weight로 입력하는 Stage 2 추상화일 뿐이다.
주소 bit-level interleaving/hash, HBM bank/channel conflict, 실제 address trace mapping은
여전히 구현하지 않으며 Stage 4 영역으로 남긴다.

Materialized H2H 경로는 source-HBM read, H2H 전송, destination-HBM write, GPU 재-read를
모델링한다. Cut-through는 source-HBM read와 H2H 전송만 수행하고 destination L2/NoC
ingress 도착을 완료로 간주한다.

materialized endpoint cost models receive-side write and subsequent reread as
separate physical accesses; this accounting has not yet been independently
validated against a detailed memory-system model.

### D-1 ring-round and side-HBM path semantics

Each attention/MLP logical collective is modeled as `R = 2*(TP-1)` globally
barriered H2H ring rounds: `TP-1` reduce-scatter rounds followed by `TP-1`
all-gather rounds. The existing per-rank aggregate ring volume is divided over
those rounds; it is never multiplied by `R` again. For a hybrid route,
`B_H2H,round = ((1-r) * B_AR) / R`, so `B_round = M/TP` only when `r=0`.
Consequently, aggregate bytes are conserved while H2H per-round latency and
queueing/dependency costs occur `R` times. `TP=1` has no collective rounds.

The GPU-direct fraction remains a single topology-abstract aggregate event.
For `0 < r < 1`, a logical collective completes only when both the explicit
H2H-round route and the aggregate direct route complete. Their different
granularities are an intentional abstraction boundary, not an implicit claim
that both links execute the same collective algorithm.

Materialized H2H endpoints are not replicated per round. Each logical
collective performs one source-HBM read, transports round chunks, and then
writes and rereads the final result from the requester's own local HBM. The
retired next-ring-tile destination is used only for selecting physical ring
edges, never as the final result owner.

`side_hbm_path=shared_h2h` is the production default. A remote read is assigned
deterministically to
`owner=(requester+1+request_index%(TP-1))%TP`; the owner's side-HBM services the
read and the large data payload traverses the directed ring from owner to
requester. Small requester-to-owner request/control bytes are omitted, making
this an optimistic control-overhead approximation. Side traffic and collective
traffic therefore contend on the same `h2h_link` edge resources.
`side_hbm_path=independent` retains the legacy separate crossing path as a
research comparison. Under `shared_h2h`, `side_hbm_link_energy_j` is zero and
the actually traversed side-read edge bytes are charged once through
`h2h_energy_j`; `collective_h2h_edge_bytes` and
`side_hbm_read_h2h_edge_bytes` expose the two byte sources separately.

### D-3 pillar-bypass side-memory path

`side_hbm_path=pillar_bypass` is a D-3 research path, not a replacement for
the production `shared_h2h` default. It models a finite package/RDL bridge that
bypasses the collective H2H ring. A remote read executes in event order as
`side_hbm[owner] service -> pillar_link[(owner, requester)] transport ->
requester dependency completion`. The large data payload travels
owner-to-requester; small request/control traffic is omitted in this v1 model.
The two directed resources `(A,B)` and `(B,A)` are distinct, so the link model
is full-duplex by direction.

Ownership and transport are deliberately separate. `side_hbm_owner_scope` is
`all_remote` for the historical pooled-owner rule or `adjacent` for an explicit
physical-neighbor graph. Adjacent ownership uses deterministic round robin,
`owner_i = physical_adjacency[tile][i % degree(tile)]`. Every TP-specific graph
must be supplied explicitly; the simulator does not invent an adjacent ring.
This permits controlled `shared_h2h+adjacent`, `independent+adjacent`, and
`pillar_bypass+adjacent` comparisons without conflating locality with transport
contention. At TP=2, all-remote and adjacent identify the same sole remote rank,
so that case isolates contention removal without an ownership-locality change.

The dedicated config is `configs/stage2_d3_pillar_bypass.yaml`. Its
conservative/nominal/aggressive entries are three paired `(BW, latency, energy)`
points, not a Cartesian product. All are pre-calibration placeholders pending
Stage 3 HFSS/ADS; there is no `N_signal -> BW` formula, `N_power` throttling,
PDN/IR-drop, EM, or thermal model in Stage 2. The qualitative trade-off that
more signal TSV area can reduce power-delivery margin is recorded only as a
hardware-design caveat.

No existing production YAML selects this path, and those configs remain
unchanged. D-3 targets a demonstrated core-model gap rather than decorative
detail: in the corrected shared-fabric capacity study, the 468 eligible decode
points at 0.02--0.50x footprint contained no B2-over-B0 wins. Separating side
reads from collective H2H contention directly tests how much of that observed
penalty is recovered by a finite bypass transport; it does not claim physical
fabricability or quantify the accompanying PDN-margin cost.

Transport energy is mutually exclusive across the three paths. Independent
uses `side_hbm_link_energy_j`, shared-H2H uses the traversed H2H edge bytes in
`h2h_energy_j`, and pillar bypass uses `pillar_link_energy_j`. The side-HBM
datapath itself is still physically accessed in every case, so
`side_hbm_energy_j` remains charged for all three. `pillar_side_read_edge_bytes`
records actual directed pillar payload bytes. Current `h2h_feasibility()` does
not include side-memory traffic or the pillar resource, so architecture
feasibility is marked not applicable for pillar-path conclusions rather than
being interpreted as “no architectural effect.”

The D-3 analysis uses

```text
recovery_fraction = (T_shared - T_pillar) / (T_shared - T_independent)
```

where 0 means shared-like performance, 1 reaches the independent reference,
values above 1 exceed it, and negative values regress relative to shared. It is
reported as null when side traffic is zero or the denominator is numerically
near zero. The analysis reuses the original eligibility rule exactly: decode
workloads with feasible B0, crossed with the 12 capacity multipliers from 0.02
through 0.50. In the corrected reference dataset this is 39 workloads x 12 =
468 eligible points; expanded pillar operating-point runs are not counted as
additional members of that 468-point population.

```powershell
.\.venv\Scripts\python.exe analysis_d3_pillar_bypass.py
```

This writes separate production-owner, controlled-adjacent, and TP=2-isolation
tables (`results/d3_pillar_bypass_production.csv`,
`results/d3_pillar_bypass_controlled_adjacent.csv`, and
`results/d3_pillar_bypass_tp2_contention_isolation.csv`) plus
`results/d3_pillar_bypass_summary.csv`. The internal
`d3_pillar_bypass_checkpoint_rows.csv` supports workload-level restart and is
not the presentation table; baseline and controlled comparisons remain
separate in the published CSVs.

`architecture_feasibility_applicable=False` for `shared_h2h`: the analytical
`h2h_feasibility()` constraint does not yet include side-HBM edge traffic. This
flag means the analytical indicator is structurally incomplete for that row,
not that the side path has no architectural impact.

Stage 1 now separates `L_DIRECT_COLLECTIVE_0` (once per topology-abstract
direct collective) from `L_H2H_ROUND_0` (once per explicit H2H ring round).
Before D-1, `K_L` scaled one H2H logical-collective latency event; after D-1 it
scales the per-round H2H latency anchor, so a logical collective incurs
`2*(TP-1)` scaled latency terms. `K_L` remains a literature/scenario
placeholder pending Stage 3 HFSS/ADS and base-die calibration. For CSV schema
compatibility, `required_h2h_latency_s` keeps its old name but now denotes the
maximum allowed **per-round** H2H latency.

각 workload step은 layer-derived stage로 나뉜다. Stage마다 memory가 준비된 뒤 producer
compute가 실행되고, 그 뒤에만 communication을 시작한다. 이후 stage의 memory/compute는
이미 생산된 이전 communication과 겹칠 수 있다. 보고되는 `overlap_s`는 동일한 signature와
parameters를 `allow_overlap=False`로 다시 실행한 순차 시간과 overlap-enabled 실행 시간의
차이인 counterfactual saved-time metric, 즉
`T_sequential - T_overlap_enabled`다. 특정 resource 구간들이 실제로 겹친 wall-clock 시간을
직접 측정한 resource-level overlap 값이 아니며, 어떤 resource가 얼마나 겹쳤는지도 식별하지
않는다.

Llama workload의 160개 stage는 attention/MLP logical collective boundary에 맞춰
semantic하게 교대로 정렬되지만, 실제 CUDA kernel 실행 단위나 kernel-level scheduling
trace는 아니다. Attention/MLP 내부 kernel, fusion, stream scheduling은 계속 추상화되어
있으므로 논문과 결과 해석에서도 kernel trace로 간주하지 않는다.

`strict_collective_dependency=True`가 production 기본값이다. 따라서 stage `i`의 memory
prefetch는 이전 communication과 겹칠 수 있지만, dependent compute는 memory와 이전
collective가 모두 끝난 뒤 시작한다. 이는 측정된 penalty의 크기 때문이 아니라 Transformer
compute가 구조적으로 선행 collective 결과에 의존한다는 causal correctness에 근거한
기본값이다. 승격 이전 loose DAG는 `strict_collective_dependency=False`를 명시하면
진단·비교 목적으로 계속 재현할 수 있다.

다음 causality audit 수치는 **Llama-70B decode, `seq_len=4096`, `r_remote=0.1` 및 아래 세
`(batch, TP)` 지점에서만** 관측한 결과이며 일반 법칙으로 확대하지 않는다.

| (batch, TP) | loose r=0 | loose r=1 | strict r=0 | strict r=1 | strict 증가(r=0) | r 효과(strict) |
|---|---:|---:|---:|---:|---:|---:|
| (1, 4) | 6.660882 ms | 6.660820 ms | 6.678929 ms | 6.660820 ms | 18.05 us (0.27%) | 18.11 us |
| (8, 4) | 7.349378 ms | 7.348921 ms | 7.465928 ms | 7.348921 ms | 116.55 us (1.59%) | 117.01 us |
| (8, 8) | 3.675173 ms | 3.674641 ms | 3.810487 ms | 3.674641 ms | 135.31 us (3.68%) | 135.85 us |

이 제한된 audit regime에서는 strict가 loose 대비 `elapsed_s`를 0.27%--3.68% 늘렸고,
노출된 dependency cost는 H2H-materialized(`r=0`) 경로에 집중됐다. 세 지점에서
strict/loose `r=1` 값이 일치한 것은 해당 조건에서 GPU-direct 완료가 다음 compute의
limiting predecessor가 아니었다는 뜻일 뿐, 다른 조건에서도 항상 같다는 뜻은 아니다.
또한 이 승격은 causal DAG만 교정한다. `r`의 byte-striping semantics와 single-server FCFS
link contention 등 communication resource fidelity 문제는 검증 완료로 간주하지 않으며
별도 backlog로 남아 있다.

`communication_time_s`는 링크가 순수하게 데이터를 전송한 시간만 뜻하지 않는다.
Communication process가 살아 있는 wall-clock interval의 union으로, materialized H2H의
endpoint local-HBM read/write/reread 대기·처리 시간, link service, queueing 및 fixed
latency를 모두 포함한다. 따라서 이 값이 크다고 해서 그만큼 critical path인 `elapsed_s`가
증가했다는 뜻은 아니다. Communication과 `elapsed_s`가 실제로 얼마나 겹치는지는 별도의
perfect-link/perfect-communication counterfactual 분석으로 확인해야 한다.

## 출력

CSV 한 행은 `(r, r_remote, batch, local_hbm_count, lane_rate_gbps, n_pillar,
side_hbm_path)` 조합 하나다.
기본 config는 batch `[1, 4, 8, 16, 32]`를 포함해 1,200개 design point를 만든다. TTFT, TPOT,
통신 시간, overlap, 총 queueing delay와 다섯 리소스 그룹의 평균 utilization을 기록한다.
`r=0` 행에는 Stage 1 기반 required H2H bandwidth/latency와 두 하드 제약의 feasible 여부도
기록한다. `r>0`은 GPU-direct fallback이 있으므로 해당 필드를 비워 둔다.
H2H 대역폭은 현재 다음 placeholder로 계산한다.

```text
BW_H2H = lane_rate * lane_count / 8 * (1 - n_pillar / n_pillar_max)
```

`n_pillar`가 커질수록 `BW_H2H`가 감소한다. 이 식은 자원 경쟁 constraint이며 PDN/IR
drop 또는 EM 시뮬레이션이 아니다.

## 채널 에너지

동적 채널 에너지는 utilization에서 역산하지 않고, SimPy `RateResource`가 기록한 실제
`transferred_amount` 바이트를 합산해 다음 식으로 계산한다.

```text
energy [J] = transferred bytes * 8 * energy [pJ/bit] * 1e-12
```

- `local_hbm`, `side_hbm` datapath: 4.33 pJ/bit. Chatterjee et al.,
  ["Architecting an Energy-Efficient DRAM System For GPUs," HPCA 2017](https://d1qx31qr3h6wln.cloudfront.net/publications/HPCA_2017_Subchannels.pdf)의
  HBM datapath 50% toggle 조건(1.48 + 2.31 + 0.54 pJ/bit)을 사용한다. 공정 세대와 실제
  uHBM 회로 차이는 아직 보정하지 않았다. 이 수치는 워크로드의 `P_total/BW`로 얻은
  effective energy가 아니라 매 transfer의 capacitance/toggle로 계산한 incremental
  datapath 에너지다. 따라서 아래 시간 기반 idle/background 항과 정적 전력을
  이중계산하지 않는다.
- `side_hbm` crossing link: 기본 10.0 pJ/bit. Arunkumar et al.,
  ["Understanding the Future of Energy Efficiency in Multi-Module GPUs," HPCA
  2019](https://d1qx31qr3h6wln.cloudfront.net/publications/HPCA_2019_EnergyEfficientGPUs_paper_0.pdf)의
  on-board integration reference다. 같은 논문의 on-package reference는 0.54 pJ/bit이며,
  32-GPM 분석에서 10/20/40 pJ/bit sensitivity를 사용했다. 실제 uHBM side-HBM 연결이
  on-package short-reach인지 full PCB/board crossing인지는 아직 확정되지 않았으므로
  10.0은 실측 uHBM 값이 아니라 nominal anchor다. `independent` 경로에서만 이 항은
  `side_hbm` datapath 4.33을 대체하지 않고 같은 실제 side 바이트의 별도 물리 요소로
  한 번 추가된다. `shared_h2h`에서는 이 항을 0으로 두고 실제 traversed edge bytes를
  아래 `h2h_link` 에너지에만 과금해 crossing energy의 이중계산을 피한다.
- `h2h_link`: UCIe 1.0 Advanced Package spec의 0.5 V 2단계 값을 사용한다.
  lane rate 12 GT/s처럼 `<16 GT/s`이면 0.25 pJ/bit, 24/32 GT/s처럼 `>=16 GT/s`이면
  0.30 pJ/bit다. 출처는 [UCIe 1.0 specification](https://www.uciexpress.org/specifications)을
  Chiu et al., [Nature Electronics 2024](https://www.nature.com/articles/s41928-024-01126-y)가
  재인용한 값이다.
- `gpu_direct_link`: Arunkumar et al.,
  ["MCM-GPU," ISCA 2017](https://research.nvidia.com/sites/default/files/publications/ISCA_2017_MCMGPU.pdf)의
  28 nm, 20 Gb/s GRS signaling 값인 0.54 pJ/bit를 사용한다.

Local-HBM stack의 nominal idle/background 항은 다음 식으로 별도 계산한다.

```text
local_hbm_idle_energy_j = TP
                        * local_hbm_stack_count
                        * hbm_idle_power_w_per_stack
                        * simulation_elapsed_s
```

`local_hbm_idle_energy_j`는 이제 TP개 GPU 전체의 idle 전력 합산치(system-wide)이며,
다른 다섯 에너지 항과 accounting 범위가 통일되어 있다. 현재 system GPU 수는 `TP`로
계산한다. `PP=1`이 고정 가정이므로 `TP * PP == TP`이기 때문이다. 이후 pipeline
parallelism(`PP>1`)을 실제로 구현하면 system GPU count를 `TP * PP`로 다시 정의하고 이
idle-energy 식도 함께 갱신해야 한다.

기본 `hbm_idle_power_w=0.4` W/stack은 Stow et al., ["Power Profiling of Modern
Die-Stacked Memory," IEEE CAL 2019](https://doi.org/10.1109/LCA.2019.2913885)의
Fig. 3(a)에 직접 보고된 4-Hi HBM2 4 GB @ 1000 MHz의 `Idle` 상태 막대를 0.2 W
gridline에서 시각 판독한 근사치다. 논문 본문에 정확한 scalar 표가 없으므로 약 ±0.1 W의
판독 불확실성이 있다. `Idle`은 clock이 켜졌지만 access가 없는 상태이며, 이 시뮬레이터가
HBM self-refresh/power-down 전이를 모델링하지 않고 모든 설정 stack을 전체 실행 시간 동안
powered-and-ready로 두기 때문에 가장 가까운 상태로 선택했다. 이 항은 순수 transistor
leakage만을 뜻하지 않으므로 필드 이름도 `hbm_idle_power_w`로 유지한다.

Sensitivity의 5.2 W stress point는 Stow의 max-bandwidth total 15.6 W/stack에 Larimi et
al., ["Understanding Power Consumption and Reliability of High-Bandwidth Memory
with Voltage Underscaling," DATE 2021](https://people.inf.ethz.ch/omutlu/pub/HBM-undervolting_date21.pdf)이
다른 실제 플랫폼에서 측정한 idle/full-load 비율 약 1/3을 곱한 cross-methodology high
reference다. 물리적 upper bound가 아니다. Stow의 직접 Idle 막대와 약 13배 차이는
platform, measurement boundary, workload/state 정의, PHY 포함 여부, 모델링 방법론 등이
후보지만 현재 자료는 지배 원인을 분리하지 못한다. 2.0 W는 두 참고점 사이의 중간
민감도 점이다.

CSV는 동적 `local_hbm_energy_j`, `side_hbm_energy_j`, `h2h_energy_j`,
`gpu_direct_energy_j`와 별도 `local_hbm_idle_energy_j`,
`side_hbm_link_energy_j`, 그리고 이 여섯 항의 `total_energy_j`를 기록한다.
capacity-infeasible 행에서는 모두 비어 있다. GPU stall/idle power, topology/hop별 에너지,
HBM power-state 전이, stack-height별 전력 scaling, PDN/IR drop/EM 및 실제 열 거동은 이번
Stage 2 에너지 모델 범위에 포함하지 않는다. 특히 Stow가 4-Hi에서 8-Hi로 static component가
약 1.4배 증가한다고 보고하지만 이번 라운드에서는 특정 Idle 막대의 정확한 scalar를
추출하거나 적용하지 않는다.

## Batch와 throughput

`workload.batch`는 스칼라 또는 리스트를 받는다. 스칼라는 하위호환을 위해 singleton
sweep으로 변환되고, 기본 config는 다음처럼 batch를 독립 축으로 탐색한다.

```yaml
workload:
  batch: [1, 4, 8, 16, 32]
```

Batched decode의 local-memory traffic은 `weight_bytes + kv_bytes * batch`다. 같은 batch
안에서 weight는 재사용하지만 요청별 KV cache는 batch에 비례한다. 이 모델은 Pope et al.,
["Efficiently Scaling Transformer Inference," 2022](https://arxiv.org/abs/2211.05102)의
batched inference memory 분석을 따른다. Tensor-parallel activation collective의
`b_remote_bytes`와 실제 연산량인 `compute_flops`도 batch에 선형 비례한다. batch=1은
기존 식과 수치적으로 완전히 동일하다.

처리량과 토큰당 에너지는 phase별로 다음처럼 기록한다.

```text
decode_throughput_tokens_per_s  = batch / tpot_s
prefill_throughput_tokens_per_s = batch * seq_len / ttft_s
decode energy_per_token_j       = total_energy_j / batch
prefill energy_per_token_j      = total_energy_j / (batch * seq_len)
```

해당하지 않는 phase의 throughput 컬럼과 capacity-infeasible 행의 모든 파생 지표는 비어
있다.

## 결과 파일 재현성

현재 config/코드로 재현하는 활성 CSV는 `stage2_materialized.csv`,
`stage2_breakeven_map.csv`, `stage2_sensitivity.csv`다.
새 side-link 및 local-stack idle/background 항으로 총 에너지와 토큰당 에너지 계산이
달라졌으므로 기존 활성 CSV는 새 스키마와 계산식으로 재생성해야 한다. 이번 코드 변경에서는
결과 CSV를 자동 재생성하지 않는다.
`stage2_pre_capacity_baseline.csv`와 `stage2_pre_stack_baseline.csv`는 생성 config가 더 이상
존재하지 않는 변경 전 비교 스냅샷이므로 새 CSV 스키마로 덮어쓰지 않고 보존한다.

승격 이전 loose-causality 결과는 `results/legacy_pre_strict_causality/`에 보존돼 있다.
이번 기본값 승격 작업은 CSV 재생성을 포함하지 않는다. Strict-default 결과가 별도로 재생성된
이후에는 `results/` 아래의 현재 production CSV를 strict-causality 결과로 해석해야 하며,
승격 이전 결과는 위 legacy 디렉터리에서 계속 확인할 수 있다.

Link/access latency는 완료 시각에는 더해지지만 serialization bandwidth resource를 계속
점유하지 않는다. Materialized 수신 capacity는 누적 전송 바이트가 아니라 flow-control로
제한된 최대 message chunk 하나를 resident working set에 더해 검사한다.

## Prefill attention 및 activation 근사

Prefill의 GPU당 compute traffic signature는 기존 linear/MLP 항에 attention의
quadratic 항을 더한다.

```text
linear_flops_per_gpu = 2 * n_params * batch * seq_len / TP
attention_flops_per_gpu
    = c_causal * 4 * n_layers * batch * seq_len^2 * d_model / TP
```

Attention의 `4`는 각 layer에서 `QK^T`와 `softmax(QK^T) * V` 두 연산을 수행하고
각 multiply-add를 2 FLOP로 세는 dense `S x S` 참조식이다. Attention head를 TP rank에
나누는 `/TP`는 Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter
Language Models Using Model Parallelism"의 head-split 구조를 따른다.
`c_causal=1.0`은 dense/naive-kernel 참조점이지 검증된 참값이 아니며,
`c_causal~=0.5`는 이상적인 causal-triangular 참조점이다. 실제 값은 kernel의 masking 및
block-skip 구현에 따라 달라지므로 분석 스크립트가 이 함수 인자를 직접 sweep한다.

Prefill activation은 resident state인 footprint와 실제 전송량인 traffic을 절대 합치지
않고 다음 두 독립 함수와 계수로 표현한다.

```text
activation_footprint_bytes
    = c_act_footprint * batch * seq_len * d_model * bytes_fp16
activation_traffic_bytes
    = c_act_traffic * batch * seq_len * d_model * bytes_fp16
```

두 식에는 구조적인 `/TP`를 적용하지 않는다. residual stream과 일부 intermediate tensor는
TP rank마다 복제될 수 있으므로, 각 effective coefficient가 TP tensor layout, fusion 및
동시에 살아 있는 intermediate tensor 수 같은 구현 의존 효과를 흡수한다. 두 계수는 문헌으로
확정된 값이 아니며 `1.0`도 normalized single-hidden-state 참조점일 뿐이다. 기본값은 둘 다
`0.0`이다. `activation_traffic_bytes()`는 prefill `b_local_bytes`에만 더하고,
`activation_footprint_bytes()`는 prefill `local_footprint_bytes`에만 더한다. 따라서
`c_act_footprint`가 0이 아니면 fixed-fraction 및 capacity-overflow placement의 resident
capacity 계산에 반영되지만, `c_act_traffic`와는 독립적으로 유지된다.

이를 보장하기 위해 `TrafficSignature.local_footprint_bytes`는 resident capacity 전용이고
`b_local_bytes`는 전송 traffic 전용이다. Capacity는
fixed-fraction capacity는
`(1-r_remote) * local_footprint_bytes + materialized_receive_buffer`로 검사한다. 기존/custom
signature에서 footprint가 `None`이면 이전 동작과의 하위 호환을 위해 `b_local_bytes`로
fallback한다. Capacity-overflow에서는 placement 이후 실제 local resident footprint와
materialized receive buffer만 downstream per-stack capacity와 비교한다. 따라서
**resident footprint와 transferred traffic은 서로 다른 회계량**이라는 불변식을 유지한다.

신규 계수는 이번 단계에서 Python 함수 인자로만 노출하며 main YAML Cartesian sweep이나 CSV
schema는 바꾸지 않는다. Decode compute 식은 이번 범위에서 그대로 둔다. 이는 physical
attention compute가 0이라는 뜻이 아니다. Long-context decode의 KV-cache attention FLOPs는
token당 `O(seq_len)`으로 증가하지만 아직 모델링하지 않은 알려진 근사다.

## 현재 placeholder와 해석 한계

2026-09-03 업데이트: 아래 항목 중 H2H lane rate, GPU-direct 대역폭, GPU peak FLOPs는
공개 문헌 기반 값으로 교체했다. 나머지는 이 프로젝트 고유의 미확정 설계값이라 공개
문헌으로 특정할 수 없어 기존 값을 유지하되 근거 없음을 명시했다.

- H2H lane rate: `[12, 24, 32]` GT/s (기존 `[13, 32, 48]`에서 교체). UCIe 1.0/1.1
  Advanced Package의 공식 data rate <-> bump pitch 페어링: 12 GT/s@25um,
  24 GT/s@36um, 32 GT/s@45um (Chiu et al., "High-performance, power-efficient
  three-dimensional system-in-package designs with universal chiplet interconnect
  express," Nature Electronics, 2024). 기존 13/48은 이 표에 없는 값이었다 -- 13은
  12의 오기로 추정되고, 48은 UCIe 3.0(2025년 발표, 2D/2.5D 전용)의 신규 rate라 세대가
  다르다. 필요하면 48/64 GT/s를 "차세대" sweep point로 별도 추가하는 것을 검토할 수 있다.
- H2H lane 수: 128 (유지). UCIe Advanced Package 모듈 폭(x64 lane/module) x 2모듈로
  해석 가능하나, 실제 설계에서 몇 모듈을 쓸지는 미정.
- `n_pillar_max`: 64 (유지, 근거 없음). 공개 문헌으로 이 프로젝트의 base die
  shoreline/via 면적 예산을 특정할 수 없었다 -- HBM base die의 정확한 floorplan이
  비공개이기 때문. Stage 3 floorplan이 나오면 교체할 것.
- GPU-direct: 768 GB/s (기존 900 GB/s에서 교체), 20 ns (유지, `TBD` 입력 시). 768 GB/s는
  MCM-GPU(Arunkumar et al., "MCM-GPU: Multi-Chip-Module GPUs for Continued
  Performance Scalability," ISCA 2017)가 on-package GPU 모듈간 직접 링크로 선택한
  "low-effort, low-cost, low-energy" 설계점이다. latency는 같은 논문의 32-cycle
  hop latency와 자릿수는 맞지만(~1.6GHz 가정 시 20ns), 정확한 클럭을 확인하지 못해
  기존 값을 그대로 유지했다.
- GPU effective peak: 990 TFLOPS (기존 1 PFLOP/s에서 교체). NVIDIA H100 SXM5 FP16
  Tensor Core dense peak (NVIDIA H100 Tensor Core GPU 공식 datasheet). 실제 target
  GPU가 H100이 아니면(예: B200/H200) 해당 스펙으로 다시 교체해야 한다.
- base-die endpoint latency: 5 ns (유지, 근거 없음). base-die endpoint 회로가 아직
  없어 공개 문헌으로 특정 불가 -- UCIe Advanced Package급 D2D PHY latency가 통상 수 ns
  수준이라는 것과 자릿수만 맞다.
- thermal 성능 배수: no=1.0, mild=0.85, severe=0.65 (유지, 근거 없음). 아직 제작되지
  않은 uHBM 패키지의 thermal 시뮬레이션 결과가 없어 공개 문헌으로 특정 불가 -- spec
  5장 의도대로 "측정값"이 아니라 sensitivity 시나리오 파라미터로 유지한다.
- Semantic Llama workload의 `N_side`는 placement 이후 side traffic이 실제로
  0보다 큰 attention/MLP stage 수에서 파생된다. Legacy/custom aggregate signature만
  기존 `n_side_requests` scheduling hint를 유지하며, 실제 요청 granularity는 향후
  workload trace/calibration이 필요하다.
- GPU당 local HBM은 개별 stack 자원으로 분리되지만, stack 내부의 4-high die 및
  bank/channel timing은 모델링하지 않는다.
- Prefill local-memory traffic은 Stage 1 weight/KV 식을 초기 근사로 사용하고, 명시적으로
  선택한 `c_act_traffic`에 따른 activation traffic을 별도로 더한다.
- `pp=1`만 지원한다. Pipeline-parallel scheduling 식이 정해질 때까지 `pp>1`은 오류다.
- `baseline`은 현재 결과 라벨이다. B0/B1/B2의 수치 preset 정의가 제공되기 전까지 실제
  동작은 `r`, `comm_mode`와 channel 파라미터가 결정한다.

위 값은 정확한 물리 예측값이 아니다. H2H/PDN 관련 값은 Stage 3 결과로, packet-level 및
GPU 동작 보정값은 Stage 4 결과로 교체해야 한다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Stage 1 일치 검사는 `r=0`, `r_remote=0`, no-contention/no-overlap 조건에서 유한한
`BW_H2H`를 스윕하고 이분 탐색으로 empirical break-even을 찾는다. Materialized endpoint
추가 비용은 이 비교에서만 제외하여 Stage 1의 end-to-end D2D 추상화와 같은 조건으로
맞춘다.
