from dataclasses import replace

import pytest

from uhbm_simulator.simulator import (
    LocalHBMCapacityError,
    SimulationParameters,
    run_simulation,
)
from uhbm_simulator.workload import TrafficSignature


def parameters(**overrides):
    base = SimulationParameters(
        tp=1,
        r=0.0,
        r_remote=0.0,
        comm_mode="materialized",
        thermal_scenario="no_throttling",
        local_hbm_stack_count=4,
        local_hbm_stack_bw_bps=25.0,
        local_hbm_stack_capacity_bytes=1_000.0,
        local_hbm_mapping_policy="uniform",
        local_hbm_mapping_weights=None,
        side_hbm_bw_bps=100.0,
        side_hbm_latency_s=0.0,
        h2h_bw_bps=1e12,
        h2h_latency_s=0.0,
        gpu_direct_bw_bps=100.0,
        gpu_direct_latency_s=0.0,
        gpu_peak_flops=1e12,
        base_die_latency_s=0.0,
        allow_overlap=False,
    )
    return replace(base, **overrides)


def local_signature(amount=100.0):
    return TrafficSignature(
        phase="decode",
        b_local_bytes=amount,
        b_remote_bytes=0.0,
        n_messages=0,
        compute_flops=0.0,
    )


def local_metrics(result):
    return {
        name: metric
        for name, metric in result.resource_metrics.items()
        if name.startswith("local_hbm")
    }


def test_four_uniform_stacks_exactly_match_one_aggregate_resource():
    aggregate = run_simulation(
        local_signature(),
        parameters(local_hbm_stack_count=1, local_hbm_stack_bw_bps=100.0),
    )
    four_stacks = run_simulation(local_signature(), parameters())

    assert four_stacks.ttft_s == aggregate.ttft_s
    assert four_stacks.tpot_s == aggregate.tpot_s


def test_more_uniform_stacks_reduce_max_per_stack_queueing():
    signature = TrafficSignature(
        phase="decode",
        b_local_bytes=0.0,
        b_remote_bytes=400.0,
        n_messages=4,
        compute_flops=0.0,
    )
    one_stack = run_simulation(
        signature,
        parameters(
            tp=2,
            local_hbm_stack_count=1,
            local_hbm_stack_bw_bps=100.0,
            allow_overlap=True,
        ),
    )
    four_stacks = run_simulation(
        signature,
        parameters(tp=2, local_hbm_stack_bw_bps=100.0, allow_overlap=True),
    )

    one_stack_max_queue = max(
        metric.total_queue_time_s for metric in local_metrics(one_stack).values()
    )
    four_stack_max_queue = max(
        metric.total_queue_time_s for metric in local_metrics(four_stacks).values()
    )
    assert one_stack_max_queue > 0.0
    assert four_stack_max_queue < one_stack_max_queue


def test_weighted_policy_creates_a_visible_stack_utilization_hotspot():
    result = run_simulation(
        local_signature(400.0),
        parameters(
            local_hbm_stack_bw_bps=100.0,
            local_hbm_mapping_policy="weighted",
            local_hbm_mapping_weights=(0.7, 0.1, 0.1, 0.1),
        ),
    )
    metrics = local_metrics(result)

    hotspot = metrics["local_hbm[0][0]"].utilization
    other_utilizations = [
        metrics[f"local_hbm[0][{stack}]"].utilization for stack in (1, 2, 3)
    ]
    assert all(hotspot > utilization for utilization in other_utilizations)


def test_weighted_policy_detects_single_stack_capacity_overflow():
    uniform = run_simulation(
        local_signature(100.0),
        parameters(local_hbm_stack_capacity_bytes=30.0),
    )
    assert uniform.tpot_s > 0.0

    with pytest.raises(LocalHBMCapacityError, match=r"stack 0.*capacity") as error:
        run_simulation(
            local_signature(100.0),
            parameters(
                local_hbm_stack_capacity_bytes=30.0,
                local_hbm_mapping_policy="weighted",
                local_hbm_mapping_weights=(0.7, 0.1, 0.1, 0.1),
            ),
        )

    assert isinstance(error.value, ValueError)
    assert error.value.stack_index == 0


@pytest.mark.parametrize(
    "weights",
    [
        (float("nan"), 1.0, 1.0, 1.0),
        (float("inf"), 1.0, 1.0, 1.0),
    ],
)
def test_direct_parameters_reject_nonfinite_mapping_weights(weights):
    with pytest.raises(ValueError, match="finite"):
        run_simulation(
            local_signature(),
            parameters(
                local_hbm_mapping_policy="weighted",
                local_hbm_mapping_weights=weights,
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("local_hbm_stack_bw_bps", float("nan")),
        ("local_hbm_stack_bw_bps", float("inf")),
        ("local_hbm_stack_capacity_bytes", float("nan")),
        ("local_hbm_stack_capacity_bytes", float("inf")),
    ],
)
def test_direct_parameters_reject_nonfinite_stack_properties(field, value):
    with pytest.raises(ValueError, match="finite"):
        run_simulation(local_signature(), parameters(**{field: value}))


def test_large_finite_weights_are_normalized_without_sum_overflow():
    result = run_simulation(
        local_signature(100.0),
        parameters(
            local_hbm_mapping_policy="weighted",
            local_hbm_mapping_weights=(1e308, 1e308, 1.0, 1.0),
        ),
    )

    assert result.tpot_s > 0.0
    assert sum(
        metric.transferred_amount for metric in local_metrics(result).values()
    ) == pytest.approx(100.0)
