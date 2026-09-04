# uHBM Simulator — 통합 설계 스펙 v3

> v2 이후 변경점: (1) "fabric" 용어 제거 → "H2H network"/"H2H interconnect"로 통일.
> (2) GPU-GPU 직접 링크 유무를 이분법이 아니라 `r` sweep 파라미터로 승격.
> (3) B0/B1/B2 네이밍 충돌 해소 (B2-pure/B2-hybrid). (4) Memory Fabric 논문(An, Lee, Jin,
> Kim, Kim, IEEE CAL 2026, DOI 10.1109/LCA.2026.3721758) 실존 확인 완료 — 정확한 숫자로
> 인용. (5) Track B 도구를 BookSim으로 확정, BookSim이 Accel-Sim에 이미 내장된 인터커넥트
> (Intersim)라는 점 반영. ASTRA-Sim은 당장은 out of scope.

---

## 0. Track A / Track B

| | Track A (메인 thesis, 지금 우선순위) | Track B (refinement, 나중) |
|---|---|---|
| 질문 | uHBM의 local-memory gain이 HBM-centric inter-tile 통신 penalty를 언제 이기는가, GPU 직접 링크가 얼마나 필요한가(`r`) | 4-stack 내부 주소 매핑/H2H network의 packet-level 거동이 `BW_uHBM,eff`/`BW_H2H,eff`를 얼마나 바꾸는가 |
| 도구 | 이 스펙의 Python event-driven simulator (simpy) | Accel-Sim (내장 BookSim/Intersim 포함) — calibration 용도 |
| 위치 | 논문의 핵심 contribution | 부록/확장 챕터 |

Track B는 ASTRA-Sim을 쓰지 않는다 (5장 근거). BookSim은 별도 도구가 아니라 Accel-Sim
안에 이미 내장된 인터커넥트 엔진("Intersim")이므로, Track B에 진입해도 "두 개를 통합"하는
작업이 아니라 "Accel-Sim의 인터커넥트 config를 uHBM 토폴로지로 커스터마이즈"하는 작업이다.

---

## 1. 핵심 연구 질문

> uHBM의 local-memory bandwidth gain이, HBM-centric inter-tile 통신
> (GPU1 → HBM1 base die → H2H interconnect → HBM2 base die → GPU2)의 penalty를
> 어떤 workload 조건에서 상쇄하는가? 그리고 그 통신에 GPU 간 직접 링크가 어느 정도
> 비율로 필요한가?

```
T_comm = T_src + T_base-die + T_H2H + T_dst
```

- `T_src`: 송신측 GPU가 자기 로컬 HBM에서 읽는 시간
- `T_base-die`: HBM base die의 네트워크 엔드포인트 처리 오버헤드 (단일 latency 상수로 추상화)
- `T_H2H`: HBM-to-HBM 링크 전송 시간
- `T_dst`: 수신측 처리 (materialized/cut-through에 따라 다름, 6장)

---

## 2. GPU 직접 링크 비율 `r` — 신규 핵심 설계 변수

이분법("GPU 직접 연결 있다/없다") 대신 연속 파라미터로 둔다:

```
r = BW_GPU-direct / BW_total-intertile     (0.0, 0.25, 0.5, 0.75, 1.0 sweep)
```

- `r = 0`: 순수 HBM-centric (모든 inter-tile 트래픽이 H2H interconnect만 경유)
- `r = 1`: 기존 GPU-centric (conventional 2.5D와 동일한 GPU-GPU 직접 통신)
- `0 < r < 1`: hybrid — GPU 직접 링크와 H2H interconnect가 트래픽을 나눠 가짐

**중요한 비대칭성**: `r=0`은 fallback이 없다. H2H 경로가 workload의 요구치를 못 채우면
그 아키텍처는 해당 workload에서 원리적으로 작동 불가능하다 (Memory Fabric 논문의 경우
GPU 직접 링크가 항상 남아있어 fallback이 있었던 것과 다름). 그래서 다음 두 하드 제약을
`r=0` 근처에서 반드시 체크한다:

```
BW_H2H,aggregate ≥ BW_required(W, TP)
L_H2H,end-to-end ≤ L_required(W, TP)
```

연구 질문: **"decode TP=N에서는 최소 몇 %의 direct 대역폭이 필요한가, 반면 MoE류
data-movement-heavy workload에서는 H2H만으로 충분한가?"**

---

## 3. 성능 모델 (Decode)

```
T_decode = T_compute + T_local-HBM + T_side-HBM + T_interGPU - T_overlap
```

| 항 | 수식 | 의미 |
|---|---|---|
| `T_local-HBM` | `B_local / BW_uHBM` | GPU 아래 4-stack 로컬 접근 (Stage 1 `b_local_per_token` 재사용) |
| `T_side-HBM` | `B_side / BW_side + N_side * L_side` | 4-stack 용량 초과분의 NUMA-like 접근 (4장) |
| `T_interGPU` | `r`에 따라 GPU-direct/H2H로 분할, 각각 `N_msg*L + B/BW` | AllReduce 등 (Stage 1 `b_remote_per_token` 재사용) |
| `T_overlap` | event simulator가 계산 | closed-form 불가 — 이게 Stage 2가 필요한 이유 |

Prefill (compute-bound):

```
T_prefill ≈ F / P_GPU,eff(thermal_scenario)
```

---

## 4. NUMA-like Side-HBM

```
r_remote = B_remote_HBM / B_total     (0, 0.1, 0.25, 0.5 sweep)
```

질문: uHBM의 local bandwidth gain이 `r_remote` 증가를 어디까지 감당하는가.

---

## 5. Thermal — "가정"이 아니라 "시나리오" (v2와 동일, 유지)

절대 "기존 구조는 무조건 throttling된다"고 단정하지 않는다. `P_GPU,eff = f(T_junction)`을
`no_throttling` / `mild_throttling` / `severe_throttling` 3개 시나리오로 sweep.

---

## 6. 두 통신 모드: Materialized (baseline) vs Cut-through (extension)

```
Mode A — Materialized (baseline, 반드시 먼저 구현)
  GPU1 → HBM1 read → H2H → HBM2 base die → HBM2 DRAM write → GPU2가 다시 read
  → 같은 랩 선행연구(An, Lee, Jin, Kim, Kim, "Memory Fabric: HBM Die-to-Die Interconnect
    for GPU Bandwidth Expansion," IEEE Computer Architecture Letters, 2026,
    DOI 10.1109/LCA.2026.3721758)의 DME "Move" 동작과 개념적으로 가장 가까움
    (source HBM → destination HBM 직접 이동, D2D payload hop count 3→1).
    단, 그 논문은 GPU-GPU 직접 링크를 유지한 채 이걸 보조 경로로 썼다는 점이
    이 문서의 r 파라미터(2장)가 필요한 이유다.

Mode B — Cut-through (extension/upper bound, materialized 검증 후에 추가)
  GPU1 → HBM1 read → H2H → HBM2 base die → GPU2 L2/NoC ingress로 직접 전달
  → 선행연구에서도 검증되지 않은 확장. GPU 마이크로아키텍처 변경 필요, 정성적으로만 다룰 것.
```

---

## 7. 물리 채널 — 여전히 HFSS/ADS(Stage 3)가 채워야 하는 gap

선행연구(Memory Fabric)조차 D2D latency를 150 cycle 고정값으로, 대역폭은 shoreline에
비례한다고 가정했을 뿐 실제 SI(insertion loss, eye diagram, BER)는 검증하지 않았다.
게다가 그건 "짧은 in-package D2D"를 전제로 한 숫자라서, uHBM처럼 base die→vertical
escape→package substrate→(경우에 따라 PCB)로 이어지는 경로에는 그대로 못 쓴다. **이게
uHBM thesis가 Memory Fabric 대비 갖는 명확한 차별점**이므로 Stage 3에서 반드시 채운다.

메시지 크기 참고값 (Llama-70B): decode TP AllReduce ~16KB/msg, small-message/high-frequency;
prefill(seq_len=4096) 수십 MB/msg, large-message/bandwidth-sensitive.

---

## 8. 시뮬레이터 아키텍처 (simpy)

```
[Workload Layer]
   (model, phase=prefill|decode, B, S, TP, PP, r, r_remote) 입력
   → Stage 1의 b_local_per_token / b_remote_per_token 재사용
      |
      v
[System/Resource Layer] (simpy)
   Resource:
     - local_hbm[tile]        capacity=BW_uHBM, 용량 한계 포함
     - side_hbm[tile]         capacity=BW_side
     - h2h_link[tile_i,j]     capacity=BW_H2H, latency=L_H2H
     - gpu_direct_link[i,j]   capacity=BW_GPU-direct  (신규 — r>0일 때만 활성)
     - gpu_compute[tile]      capacity=P_GPU,eff(thermal_scenario)
   comm_mode: materialized | cut_through
      |
      v
[Stats Layer]
   TTFT, TPOT, 리소스별 utilization, 큐잉 지연, r별/r_remote별/thermal_scenario별 비교
```

---

## 9. Config YAML

```yaml
baseline: B2-hybrid    # B0 | B1 | B2-pure | B2-hybrid  (B0/B1/B2는 기존 정의 유지, 하위에 pure/hybrid 추가)
comm_mode: materialized
thermal_scenario: no_throttling
r: [0, 0.25, 0.5, 0.75, 1.0]
channels:
  local_hbm:
    bw_bps: 4.8e12
    capacity_bytes: 64e9
  side_hbm:
    bw_bps: 4.8e12
    latency_s: 50e-9
  h2h:
    lane_rate_gbps: [13, 32, 48]   # 확정값 아님 — sweep
    latency_s: 20e-9
    n_pillar: [4, 8, 16, 32]        # 신규(13장) — PDN pillar 수, BW_H2H를 이 값의 함수로 계산
    n_pillar_max: TBD                # 신규 — shoreline 면적으로 정해지는 상한, placeholder
  gpu_direct:            # 신규
    bw_bps: TBD
    latency_s: TBD
workload:
  model: Llama-70B
  tp: 4
  pp: 1
  seq_len: 4096
  batch: 1
  r_remote: [0, 0.1, 0.25, 0.5]
```

---

## 10. 필수 검증

리소스 무한대·큐잉 없음 설정 시 Stage 1(`required_bw_d2d`)과 일치. 추가로:

- `BW_uHBM` ↑ ⇒ TPOT ↓
- `BW_H2H` ↑ ⇒ T_comm ↓ (TP-heavy)
- `L_H2H` ↑ 시 decode가 prefill보다 민감도 큼
- `r` ↑ (GPU-direct 비중 증가) ⇒ H2H 의존도 감소, `r=1`에서 conventional B0와 T_interGPU 일치

---

## 11. SCOPE LOCK

- GPU cycle-accurate 시뮬레이션 금지, 실제 GPU 트레이스 불필요
- thermal은 parameterized scenario로만, 정량 열 시뮬레이션(FEA) 금지
- PDN/IR drop은 별도 미니 스크립트, 이 시뮬레이터 안에 넣지 않음
- 주소 매핑/bank conflict 금지 — Track B로 분리
- **지금 로컬에 있는 BookSim/, Accel-Sim/, ASTRA-Sim/ 세 폴더는 Track A 구현에서 import,
  빌드, 실행 전부 금지.** Track A는 이 세 저장소와 무관하게 순수 Python(simpy)으로만 만든다.
- ASTRA-Sim은 Track B에서도 사용하지 않는다 (같은 랩 선행연구가 Accel-Sim 내장 BookSim으로
  검증한 선례가 있고, ASTRA-Sim의 collective/topology 추상화 레벨은 H2H interconnect의
  packet-level 거동을 보는 목적과 맞지 않음).

---

## 12. 용어 정리 (naming 통일)

원래 문서(uHBM_워크로드 분석 초안_v3, HFSS_ADS_실험설계_v4)의 **Stage 1~4 체계를 그대로
쓴다.** "Track A/B"는 별도 용어가 아니라 Stage 1(analytical)+Stage 2(event-driven)를
합쳐 부르는 편의상 이름일 뿐이다.

- **Stage 1 + Stage 2 (= "Track A")**: 이 스펙 문서의 대상. 지금 최우선 작업.
- **Stage 4 (= "Track B")**: Accel-Sim(내장 BookSim/Intersim) 기반 calibration. 나중.
- **Stage 3**: HFSS/ADS 물리 채널 검증 + PDN/IR drop feasibility. 원래 문서 정의 그대로,
  "Track C"라는 이름은 쓰지 않는다.

---

## 13. Addendum — PDN(Cu pillar)과 H2H 대역폭의 연결

지금까지 PDN(IR drop)과 H2H 네트워크 대역폭을 완전히 분리된 챕터로 다뤘는데, 실제로는
둘이 같은 물리적 자원(base die shoreline/via 면적)을 두고 경쟁할 수 있다. 이걸 SCOPE LOCK을
어기지 않는 선에서(EM 시뮬레이션 없이) 연결한다.

```
1) 전력 제약: V_drop(N_pillar) ≤ V_budget 을 만족하는 최소 pillar 수
       N_pillar ≥ N_pillar*     (Track A 밖의 별도 미니 스크립트, R=ρL/A 기반, 그대로 유지)

2) 자원 경쟁: base die shoreline/via 면적은 유한하므로, pillar가 더 필요할수록
   H2H 링크에 쓸 수 있는 면적/레인 수가 줄어든다.
       BW_H2H = f(N_pillar)     (단조 감소 함수 — 정확한 형태는 모름, sweep으로 대체)
```

**Track A 구현에 반영할 것**: `BW_H2H`를 완전히 독립적인 자유 파라미터로 두지 말고,
`n_pillar`라는 새 config 값을 추가해서 `BW_H2H`가 `n_pillar`에 대해 단조 감소하는
placeholder 관계(예: `BW_H2H = BW_H2H_max * (1 - n_pillar / n_pillar_max)` 같은 선형
근사, 실제 함수 형태는 TODO)로 계산되게 한다. 이러면 "전력 예산을 넉넉히 잡을수록
통신 대역폭이 희생된다"는 트레이드오프가 sweep 결과에 자연스럽게 나타난다. 정확한
`f()`는 Stage 3(HFSS/ADS + PDN)에서 검증할 대상이며, Track A에서는 sweep 가능한
placeholder로만 둔다.

---

## 14. Codex 프롬프트는 별도 파일(`codex_prompt_track_a.md`) 참고
