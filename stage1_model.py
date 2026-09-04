# -*- coding: utf-8 -*-
"""
uHBM Stage 1 -- LLM Workload -> Traffic Signature -> Break-even 해석 모델

목적: "GPU-on-HBM 수직 적층 구조가 언제 유리한가"를 물리 시뮬레이션(HFSS/ADS) 없이
     순수 파이썬 수식만으로 먼저 답해보는 스크립트.
     uHBM_워크로드 분석 주제 구체화 초안_v3.pdf 의 01/04/09장 수식을 그대로 구현했다.

파이프라인에서 이 스크립트의 위치:
    [HFSS: 형상 -> S-parameter]  -->  [ADS: S-parameter -> achievable data rate]
                                              |
                                              v  (이 값이 BW_LOCAL_0, k_B 로 들어온다)
    [이 스크립트: workload -> traffic signature -> break-even 조건]

지금 단계에서는 HFSS/ADS 숫자가 아직 없으므로, k_B(로컬 메모리 대역폭 개선 배수)를
"주어진 값"이 아니라 "스윕 변수"로 둔다. 즉 "uHBM이 이기려면 k_B가 최소 얼마나
필요한가?"를 answer하는 것이 이 파일의 역할이다 (원 문서의 Stage 1 정의 그대로).
"""

import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# 0. 모델 아키텍처 스펙 -- 이 값들은 "진짜" 숫자 (공개된 Llama 아키텍처 스펙)
# ======================================================================
MODELS = {
    "Llama-7B":  dict(n_layers=32, d_model=4096,  n_params=7e9),
    "Llama-70B": dict(n_layers=80, d_model=8192,  n_params=70e9),
}
BYTES_FP16 = 2  # weight / activation / KV cache 모두 FP16 가정 (2 bytes/element)


# ======================================================================
# 1. Local memory traffic: B_local(token) = B_weight + B_KV  (+B_act, 여기선 생략)
#    -> 워크로드 초안 04장의 B_local = B_weight + B_KV + B_activation 중
#       decode 단계에서 지배적인 두 항만 우선 구현 (B_act는 상대적으로 작음)
# ======================================================================
def b_local_per_token(model, tp, seq_len, batch=1):
    """decode 1 스텝에서, GPU 1개(TP로 쪼개진 몫)가 로컬 HBM에서 읽어야 하는 바이트."""
    n_layers, d_model, n_params = model["n_layers"], model["d_model"], model["n_params"]

    # (a) weight: decode는 memory-bound라 매 토큰마다 TP 몫 weight 전체를 한 번씩 읽는다고 가정
    b_weight = (n_params / tp) * BYTES_FP16

    # (b) KV cache: layer마다 K,V 각각 (d_model/tp) 만큼을, 지금까지 쌓인 seq_len 토큰 수만큼 읽는다
    #     -> 원 문서 지적대로, seq_len(S)이 길어질수록 이 항이 지배적이 된다 (long-context decode)
    b_kv = 2 * n_layers * (d_model / tp) * seq_len * BYTES_FP16

    return (b_weight + b_kv) * batch


# ======================================================================
# 2. Inter-tile traffic: B_remote(token), N_msg(token)
#    Megatron-style Tensor Parallel -> layer마다 attention/MLP 각 1회씩 AllReduce
#    (원 문서의 "Traffic Signature 추상화" W=(B_local,B_remote,N_msg,...) 중 두 항)
# ======================================================================
def b_remote_per_token(model, tp, batch=1):
    """TP AllReduce로 GPU 1개가 실제 주고받는 바이트 수와, 토큰당 메시지(=collective 호출) 횟수."""
    if tp == 1:
        return 0.0, 0  # TP=1이면 애초에 inter-tile 통신이 없다 -> uHBM이 항상 유리한 자명한 case

    n_layers, d_model = model["n_layers"], model["d_model"]
    n_msg = 2 * n_layers  # attention 1회 + MLP 1회, layer마다
    msg_size = batch * d_model * BYTES_FP16

    # Ring AllReduce 표준 비용식: GPU 1개가 주고받는 바이트 ~= 2*(TP-1)/TP * msg_size
    b_remote = n_msg * 2 * (tp - 1) / tp * msg_size
    return b_remote, n_msg


# ======================================================================
# 3. 물리 링크 파라미터 -- ⚠️ 지금은 "문헌 기반 placeholder"다.
#    나중에 HFSS(CH0/CH1 S-parameter) + ADS(achievable data rate)로 반드시 교체할 것.
#    지금 목적은 "정확한 숫자"가 아니라 "모델 구조가 맞는지" 확인하는 것.
# ======================================================================
BW_LOCAL_0     = 4 * 1.2e12   # [문헌] CH0 baseline: HBM3e 1.2TB/s/stack x 4 stack = 4.8TB/s
BW_INTERTILE_0 = 900e9        # [TODO: HFSS/ADS 또는 문헌으로 교체] baseline interposer GPU-GPU 대역폭
L_D2D_0        = 20e-9        # [TODO: HFSS/ADS 또는 문헌으로 교체] baseline D2D latency/hop
K_L            = 1.0          # 원 문서 정의대로 1차 설계축이 아니라 sensitivity 파라미터 (기본 1.0)


# ======================================================================
# 4. Break-even 조건 (원 문서 04/09장의 "Main Result" 수식 그대로)
#
#    T_comm,u(BW_D2D*) = T_comm,0 + T_mem,0 * (1 - 1/k_B)
#
#    풀어 쓰면: uHBM에서 로컬 메모리 접근으로 절약한 시간(T_mem,0*(1-1/k_B))이
#    "예산"이 되어, 그 예산 안에서만 inter-tile 통신 시간이 늘어나도 괜찮다.
#    이 식을 BW_D2D*에 대해 풀면 "우리 워크로드에서 uHBM이 이기려면 D2D 채널이
#    최소 얼마의 대역폭을 내야 하는가"가 나온다 -- 이것이 HFSS/ADS가 나중에
#    검증해야 할 목표 스펙(feasibility target)이 된다.
# ======================================================================
def required_bw_d2d(model, tp, seq_len, k_B, batch=1):
    b_loc = b_local_per_token(model, tp, seq_len, batch)
    b_rem, n_msg = b_remote_per_token(model, tp, batch)

    if b_rem == 0:
        return 0.0  # TP=1: inter-tile 통신이 없으므로 D2D 대역폭 요구사항 자체가 없음

    T_mem0 = b_loc / BW_LOCAL_0
    T_comm0 = n_msg * L_D2D_0 + b_rem / BW_INTERTILE_0
    L_D2D_u = K_L * L_D2D_0

    budget = T_comm0 + T_mem0 * (1 - 1 / k_B) - n_msg * L_D2D_u
    if budget <= 0:
        return np.inf  # 이 조건에서는 k_B가 아무리 커도 uHBM이 물리적으로 이길 수 없음 (failure case)
    return b_rem / budget


# ======================================================================
# 5. Sanity check -- 지도교수님/선배님께 보여줄 첫 번째 확인
#    "k_B가 커질수록 필요한 BW_D2D*는 감소해야 한다" (원 문서 06장에서 이미 증명한 부호)
# ======================================================================
def sanity_check():
    model = MODELS["Llama-70B"]
    print("=== Sanity check: k_B 증가 -> BW_D2D* 감소하는가? ===")
    for tp in (1, 2, 4, 8):
        vals = [required_bw_d2d(model, tp, seq_len=4096, k_B=kb) for kb in (1.0, 1.5, 2.0, 3.0)]
        monotonic = all(a >= b for a, b in zip(vals, vals[1:]))
        print(f"TP={tp:>2}  BW_D2D* @k_B=[1.0,1.5,2.0,3.0] (GB/s) = "
              f"{[round(v/1e9,1) if np.isfinite(v) else float('inf') for v in vals]}  "
              f"monotonic_decrease={monotonic}")
    print()


# ======================================================================
# 6. Suitability Map -- 원 문서 Fig.3의 "개념도"를 실제 숫자로 그린 버전
# ======================================================================
def plot_suitability_map():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    k_B_range = np.linspace(1.01, 3.0, 60)

    # (좌) 모델 크기 x TP 별 비교 -- seq_len=4096 고정
    ax = axes[0]
    for name, model in MODELS.items():
        for tp in (2, 4, 8):
            y = [required_bw_d2d(model, tp, seq_len=4096, k_B=kb) / 1e9 for kb in k_B_range]
            ax.plot(k_B_range, y, label=f"{name}, TP={tp}")
    ax.set_xlabel("Local memory bandwidth gain  $k_B$  (uHBM / 2.5D)")
    ax.set_ylabel("Required $BW_{D2D}^{*}$  (GB/s)")
    ax.set_title("Suitability boundary (seq_len=4096, decode)")
    ax.set_yscale("log")
    ax.axhline(BW_INTERTILE_0 / 1e9, color="gray", linestyle="--", linewidth=1,
               label="baseline $BW_{intertile,0}$ (reference)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (우) long-context 효과 -- Llama-70B, TP=4 고정, seq_len 스윕
    ax = axes[1]
    model = MODELS["Llama-70B"]
    for seq_len in (128, 1024, 4096, 16384, 65536):
        y = [required_bw_d2d(model, tp=4, seq_len=seq_len, k_B=kb) / 1e9 for kb in k_B_range]
        ax.plot(k_B_range, y, label=f"seq_len={seq_len}")
    ax.set_xlabel("Local memory bandwidth gain  $k_B$  (uHBM / 2.5D)")
    ax.set_ylabel("Required $BW_{D2D}^{*}$  (GB/s)")
    ax.set_title("Long-context decode effect (Llama-70B, TP=4)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("uHBM Suitability Map -- Stage 1 analytical model "
                  "(literature-based baseline, k_B/k_L swept)",
                  fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("suitability_map_v1.png", dpi=160)
    print("Saved: suitability_map_v1.png")


if __name__ == "__main__":
    sanity_check()
    plot_suitability_map()
