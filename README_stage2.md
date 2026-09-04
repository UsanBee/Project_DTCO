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
`n_pillar=8`이며 기본 Llama-70B 설정에서 capacity-feasible하다. `r`, `r_remote`, H2H
lane rate, `n_pillar`는 입력 config의 리스트를 사용한다. `local_hbm_count`는 config의
`1..max_count`, categorical 축은 검증된 전체 domain(materialized/cut-through,
thermal 3개 scenario, uniform/weighted)을 사용한다. 각 축 이외의 값은 모두 기준점에
고정한다.

`tpot_s_relative_range`와 `communication_time_s_relative_range`는 capacity-feasible
포인트만 사용한 `(max-min)/baseline`이다. `capacity_infeasible_fraction`과
`n_capacity_infeasible_points`는 값이 없는 포인트의 비율과 개수다.

`flips_architecture_feasible`은 항상 `h2h_feasibility()`의 **native `r=0` semantics**로
shadow 평가하며, OAT row의 실제 `r` 값과 무관하다. 따라서 `r` 축 자체에서는 이 값이
구조적으로 상수이고 flip이 없는 것이 예상 동작이지 버그가 아니다.
`r_remote`, `comm_mode`, `thermal_scenario`, `address_mapping.policy`도
`h2h_feasibility()`의 입력이 아니므로 이 지표의 flip은 구조상 항상 `False`다. 이를
"실험 결과 해당 축이 architecture에 영향 없음"으로 해석하면 안 된다. CSV의
`architecture_feasibility_applicable=False`가 이러한 비적용을 표시하고,
`local_hbm_count`, `h2h.lane_rate_gbps`, `h2h.n_pillar`만 `True`다.

`address_mapping.policy=weighted` 포인트는 `[0.7, 0.1, 0.1, 0.1]`을 사용한다. 이는
평균적인 weighted 분포가 아니라 **의도적으로 만든 worst-case/hotspot profile**이다.
기본 Llama-70B 결과에서는 이 weighted 포인트가 capacity-infeasible이므로 효과가 metric
elasticity가 아니라 `flips_capacity_feasible=True`로 나타난다. 향후 feasible한 weighted
포인트에서 큰 elasticity가 나오더라도 일반적인 weighted mapping의 효과가 아니라 hotspot
민감도로 해석해야 한다. 이 설명은 CSV의 `notes`에도 함께 기록된다. 실제 bit-level 주소
매핑이나 bank/channel conflict를 구현한다는 뜻은 아니다.

## 트래픽 분리

두 비율은 서로 독립적이다.

```text
b_local  -> (1-r_remote) local_hbm + r_remote side_hbm
b_remote -> (1-r) h2h_link + r gpu_direct_link
```

## GPU별 local-HBM stack

`local_hbm`은 GPU당 aggregate 자원 하나가 아니라 `(tile, stack_index)`로 구분된 개별
HBM stack 자원이다. 기본 구성은 GPU 아래 2x2로 배치된 HBM 4개이며, 각 HBM은 내부적으로
4-high stack이다. 기본 per-stack 대역폭/용량은 1.2 TB/s와 16 GB이므로 uniform 4-stack의
총 대역폭과 용량은 기존 모델과 동일한 4.8 TB/s와 64 GB이다.

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

**Scope lock:** `address_mapping.policy`라는 이름은 실제 주소 매핑 구현을 뜻하지 않는다.
이 기능은 트래픽 분배 결과를 declarative policy/weight로 입력하는 Stage 2 추상화일 뿐이다.
주소 bit-level interleaving/hash, HBM bank/channel conflict, 실제 address trace mapping은
여전히 구현하지 않으며 Stage 4 영역으로 남긴다.

Materialized H2H 경로는 source-HBM read, H2H 전송, destination-HBM write, GPU 재-read를
모델링한다. Cut-through는 source-HBM read와 H2H 전송만 수행하고 destination L2/NoC
ingress 도착을 완료로 간주한다.

각 workload step은 layer-derived stage로 나뉜다. Stage마다 memory가 준비된 뒤 producer
compute가 실행되고, 그 뒤에만 communication을 시작한다. 이후 stage의 memory/compute는
이미 생산된 이전 communication과 겹칠 수 있다. 보고되는 overlap은 같은 입력을 overlap
없이 실행한 시간과 causal 실행 시간의 차이다.

## 출력

CSV 한 행은 `(r, r_remote, lane_rate_gbps, n_pillar)` 조합 하나다. TTFT, TPOT,
통신 시간, overlap, 총 queueing delay와 다섯 리소스 그룹의 평균 utilization을 기록한다.
`r=0` 행에는 Stage 1 기반 required H2H bandwidth/latency와 두 하드 제약의 feasible 여부도
기록한다. `r>0`은 GPU-direct fallback이 있으므로 해당 필드를 비워 둔다.
H2H 대역폭은 현재 다음 placeholder로 계산한다.

```text
BW_H2H = lane_rate * lane_count / 8 * (1 - n_pillar / n_pillar_max)
```

`n_pillar`가 커질수록 `BW_H2H`가 감소한다. 이 식은 자원 경쟁 constraint이며 PDN/IR
drop 또는 EM 시뮬레이션이 아니다.

Link/access latency는 완료 시각에는 더해지지만 serialization bandwidth resource를 계속
점유하지 않는다. Materialized 수신 capacity는 누적 전송 바이트가 아니라 flow-control로
제한된 최대 message chunk 하나를 resident working set에 더해 검사한다.

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
- `N_side`는 현재 모델 layer 수로 근사한다. 실제 요청 수는 workload
  trace/calibration 결과가 필요하다.
- GPU당 local HBM은 개별 stack 자원으로 분리되지만, stack 내부의 4-high die 및
  bank/channel timing은 모델링하지 않는다.
- Prefill local-memory traffic은 Stage 1 decode traffic 식을 초기 근사로 사용한다.
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
