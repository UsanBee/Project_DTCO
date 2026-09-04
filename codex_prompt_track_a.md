이 폴더(Simulator/)에는 지금 4가지가 있다:

- `uhbm_simulator_spec_v3.md` — 설계 스펙 (이 프롬프트가 참조하는 문서, 반드시 먼저 전체를 읽을 것)
- `stage1_model.py` — 이미 완성된 Stage 1 analytical model (재사용 대상)
- `BookSim/`, `Accel-Sim/`, `ASTRA-Sim/` — 참고용으로만 받아둔 오픈소스 시뮬레이터 3종

먼저 지금 할 일과 하지 않을 일을 명확히 한다.

**지금 만들 것 (Track A)**: `uhbm_simulator_spec_v3.md`의 설계를 그대로 구현하는, `simpy`
기반의 순수 Python discrete-event 시뮬레이터. `stage1_model.py`의 `b_local_per_token`,
`b_remote_per_token`을 import해서 재사용한다.

**지금 하지 않을 것**: `BookSim/`, `Accel-Sim/`, `ASTRA-Sim/` 폴더의 코드를 import하거나,
빌드하거나, 서브프로세스로 실행하거나, 소스 레벨로 참조하지 않는다. 이 세 폴더는 나중에
Track B(별도 스펙 예정)에서만 쓰일 예정이고, 그마저도 셋을 다 쓰는 게 아니라 Accel-Sim
(내장된 BookSim/Intersim 인터커넥트 포함)만 쓸 계획이다. ASTRA-Sim은 이 프로젝트에서
쓰지 않기로 확정했다. 지금 코드에서는 이 세 폴더의 존재를 완전히 무시해도 된다 — 구조나
아이디어를 참고하고 싶으면 코드가 아니라 설명(문서/README)만 읽고, Python 구현에는
그대로 반영하지 마라.

**구현 순서**:

1. `uhbm_simulator_spec_v3.md`의 0~13장을 전부 읽고, 특히 다음을 정확히 지켜라:
   - 2장의 `r` 파라미터 (GPU-direct 비율) — 아키텍처를 이분법으로 만들지 말 것
   - 6장의 두 통신 모드 — `materialized`를 먼저 구현하고 검증한 뒤에만 `cut_through` 추가
   - 5장의 thermal — 반드시 `no_throttling`/`mild_throttling`/`severe_throttling` 세 개의
     시나리오 파라미터로만 다루고, 실제 열 시뮬레이션은 절대 구현하지 마라
   - 13장의 PDN-H2H 연결 — `BW_H2H`를 `n_pillar`의 단조 감소 함수로 계산 (정확한 함수
     형태는 모르니 config로 조정 가능한 선형 근사 placeholder로 구현하고 TODO 주석)
   - 11장 SCOPE LOCK 전체 — 특히 PDN/IR drop 자체를 EM 레벨로 계산하려 하지 말 것,
     13장의 관계식은 어디까지나 자원 경쟁을 나타내는 근사 constraint일 뿐이다
   - 12장의 용어 — 코드 주석/변수명/문서에서 "Track C" 같은 이름을 만들지 말고 Stage
     1/2/3/4 체계를 따를 것
2. 9장의 YAML config 포맷(n_pillar 포함) 그대로 config 파일을 읽는 로더를 만들어라.
3. 8장의 아키텍처(Workload Layer → System/Resource Layer(simpy) → Stats Layer)를 그대로
   구현하되, 리소스는 `local_hbm`, `side_hbm`, `h2h_link`, `gpu_direct_link`(r>0일 때만
   활성화), `gpu_compute` 다섯 개를 만들어라.
4. 10장의 sanity check를 전부 통과하는 테스트를 작성하고 실행 결과를 보여줘라. 특히
   `stage1_model.py`의 `required_bw_d2d`와 리소스 무한대/큐잉 없음 조건에서 일치하는지
   확인하는 테스트가 제일 중요하다 — 이게 실패하면 구현에 버그가 있는 것이다. 여기에
   하나 추가: `n_pillar` ↑ ⇒ `BW_H2H` ↓ (13장 관계가 sweep 결과에 반영되는지)도 확인해라.
5. 마지막으로, 스펙 문서 어딘가에서 결정이 안 난 부분(예: `gpu_direct`의 실제 BW/latency
   값, `h2h`의 실제 lane 수)을 코드에 `# TODO(태현):` 주석으로 명시하고, 지금은 문서에 있는
   placeholder 값이나 합리적인 기본값으로 채워서 일단 돌아가게 만들어라.

작업을 시작하기 전에, 위 내용을 어떻게 구현할지 계획을 먼저 글로 짧게 요약해서 보여주고,
내 확인을 받은 뒤에 실제 코드를 작성해줘.
