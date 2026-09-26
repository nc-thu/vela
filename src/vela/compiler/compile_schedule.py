"""Compile grouped streams and nonlinear-unit reservations.

The public paper presets use format A (W8A8 G64). Historical format helpers
remain internal so the archived schedule can be replayed without arithmetic
changes. See docs/architecture.md for the model's execution scope.
"""
from .group_schedule import build as grouped_build
from .w4a8_format import apply as apply_format, CONFIGS


def build(*args, config=None, **kwargs):
    p = grouped_build(*args, **kwargs)
    p["version"] = 7
    p["sfu"].update(
        implementation="PWL64 mixed-unit vector core",
        elementwise_chain=["convert", "mul", "add", "convert"],
        elementwise_functions=["gelu", "tanh", "sin", "cos"],
        unit_instances={
            "convert": 32,
            "mul": 32,
            "add": 32,
            "exp": 8,
            "reciprocal": 2,
            "rsqrt": 1,
            "reduce": 1,
            "sincos": 0,
        },
        values_per_cycle={
            "convert": 32,
            "mul": 32,
            "add": 32,
            "exp": 8,
            "reciprocal": 2,
            "rsqrt": 1,
            "reduce": 32,
        },
        latencies={
            "convert": 4,
            "mul": 7,
            "add": 12,
            "exp": 14,
            "reciprocal": 16,
            "rsqrt": 16,
            "reduce": 16,
        },
        pwl_segments=64,
        coefficient_bytes_per_function=256,
        pwl_table_bytes=32768,
        segment_selection="folded into convert; no separate latency, per source design",
        reduction="one 32-input tree; inherited 16-cycle timing assumption",
        precision="FP16 PWL; FP32 reductions/normalization retain inherited timing abstraction",
        evidence="Paper HNU schedule: explicit FP16 pipeline depths and service rates",
    )
    p["resources"]["pwl_LUTRAM_bytes"] = 32768
    p["resources"]["pwl_LUT_count"] = None
    p["resources"]["vector_core_DSP_count"] = None
    p["execution_contract"]["sfu_sharing"] = (
        "PWL microprogram reserves shared convert/mul/add until block drain; "
        "softmax/LN issue on the same named units at 32/32/32/8/2/1 throughput; "
        "conservative inter-block reservations, exact intra-PWL microblock schedule"
    )
    p["execution_contract"]["quality"] = (
        "PWL64 formula-level evidence only; no full numerical replay or LIBERO validation; "
        "uniform-segment fit is not a verified FP16-bit-index coefficient table"
    )
    if config is not None:
        assert config in CONFIGS
        p = apply_format(p, config)
    return p
