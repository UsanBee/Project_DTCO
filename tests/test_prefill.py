from stage1_model import MODELS
from uhbm_simulator.simulator import SimulationParameters, run_simulation
from uhbm_simulator.workload import build_prefill_signature


def test_prefill_signature_is_compute_dominant_and_scales_collective_payload():
    signature = build_prefill_signature("Llama-70B", tp=4, seq_len=4096, batch=1)

    assert signature.phase == "prefill"
    assert signature.token_count == 4096
    assert signature.compute_flops == 2 * MODELS["Llama-70B"]["n_params"] * 4096 / 4
    assert signature.b_remote_bytes > 1e9


def test_prefill_reports_ttft_and_amortized_tpot():
    signature = build_prefill_signature("Llama-7B", tp=1, seq_len=10, batch=1)
    parameters = SimulationParameters(
        tp=1,
        r=0.0,
        r_remote=0.0,
        comm_mode="materialized",
        thermal_scenario="no_throttling",
        local_hbm_stack_count=1,
        local_hbm_stack_bw_bps=1e30,
        local_hbm_stack_capacity_bytes=1e30,
        local_hbm_mapping_policy="uniform",
        local_hbm_mapping_weights=None,
        side_hbm_bw_bps=1e30,
        side_hbm_latency_s=0.0,
        h2h_bw_bps=1e30,
        h2h_latency_s=0.0,
        gpu_direct_bw_bps=1e30,
        gpu_direct_latency_s=0.0,
        gpu_peak_flops=1e12,
        base_die_latency_s=0.0,
    )

    result = run_simulation(signature, parameters)

    assert result.ttft_s == result.tpot_s * 10
