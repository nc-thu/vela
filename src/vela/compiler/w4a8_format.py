"""W4A8/G32/Pack3 format application over the v18 (v12-baseline) plan.

Five configs (comparison matrix from the handoff PLAN §5):
  A w8a8_g64_pack2  : identical to the v12 baseline; must reproduce it bit-exact
  B w8a8_g32_pack2  : isolates the G32 scale/merge cost
  C w4a8_g64_pack2  : isolates the weight-compression benefit
  D w4a8_g32_pack2  : conservative target precision
  E w4a8_g32_pack3  : full target (3 products per DSP per cycle on static GEMMs)

Static-weight GEMMs (linear/conv2d with constant weights) switch to
weight_bits/group_k/packing per config. Dynamic activation x activation GEMMs
(baddbmm, sdpa) stay A8 x A8 Pack2 in every config and are tagged as such.

Panel re-tiling: candidates {16,32,48,64,96} as in v13; for packing 3 the extra
candidate 144 is allowed (48 physical columns x 3 logical columns). 144 is not
a multiple of the 32-column output group, so those panels carry a
cross_group_carry flag; the cycle model charges the straddling output-group
re-encode and the store-overlap hazard instead of ignoring it. W8 never uses
144 (its weight feed would need 144 B/cycle > the 96 B port).

Weight storage per column: data = k bytes (W8) or ceil(k/2) nibble bytes (W4),
records = C(k,group)*8. Ideal ratio W4G32 vs W8G64 = (K/2+8K/32)/(K+8K/64) = 2/3.
"""
import math

C = lambda n, d: (n + d - 1) // d

CONFIGS = {
    "A": dict(weight_bits=8, group_k=64, packing=2),
    "B": dict(weight_bits=8, group_k=32, packing=2),
    "C": dict(weight_bits=4, group_k=64, packing=2),
    "D": dict(weight_bits=4, group_k=32, packing=2),
    "E": dict(weight_bits=4, group_k=32, packing=3),
}


def col_bytes(k, wbits, g):
    data = k if wbits == 8 else C(k, 2)
    return data + C(k, g) * 8


def apply(p, name):
    cfg = CONFIGS[name]
    wbits, g, pk = cfg["weight_bits"], cfg["group_k"], cfg["packing"]
    fmt = {
        "config": name,
        "weight_bits": wbits,
        "activation_bits": 8,
        "weight_group_k": g,
        "activation_group_k": g,
        "output_group_n": 32,
        "packing_factor_static": pk,
        "packing_factor_dynamic": 2,
        "weight_grid": "[-7,7] symmetric zero RNE (hardware bit-exact for [-8,7])"
        if wbits == 4
        else "[-127,127] symmetric zero RNE (hardware covers [-128,127])",
        "activation_grid": "[-128,127] signed INT8",
        "group_dot": "exact INT32; 17-bit signed worst case at G32 (32768)",
        "merge": "per-(row,col) common exponent, Q24 coefficients, INT64 sum",
        "nibble_order": "low nibble = even k index",
        "dynamic_gemm_policy": "A8xA8 Pack2 in every config; per-config K grouping follows group_k",
        "cross_group_carry": "required for static Pack3 144-column panels (144 % 32 != 0)",
        "reference": "compiler v19 g32_reference.py + hw v19 pack3_reference.py",
    }
    p["formats"] = fmt
    panel_capacity = 384 * 1024  # single weight bank, as in the baseline config
    static_cnt = dyn_cnt = 0
    carry_chains = 0
    for cmd in p["commands"]:
        for r in cmd["matrices"]:
            if r["constant_b"]:
                r["format"] = f"w{wbits}a8_static_pack{pk}"
                static_cnt += 1
            else:
                r["format"] = "a8a8_dynamic_pack2"
                dyn_cnt += 1
            k = r["k"]
            cands = []
            for nc in (16, 32, 48, 64, 96, 144):
                if nc == 144:
                    if pk != 3 or wbits != 4 or r["n"] <= 144:
                        continue  # Pack3-only, W4-only, and only as a real panel
                elif nc % 32 and r["n"] > nc:
                    continue  # v13 rule preserved for non-Pack3 paths
                foot = nc * col_bytes(k, 8 if not r["constant_b"] else wbits, g)
                if foot <= panel_capacity:
                    tiles = C(r["m"], 16) * C(r["n"], nc) * r["batch"]
                    est = tiles * (C(k, g) * (g + 16) + 2 * C(16 * nc, 8) + 16 * C(nc, 32) + 14)
                    cands.append({"cols": nc, "panel_bytes": foot, "array_estimate": est})
            assert cands, ("no panel fits", r, panel_capacity)
            chosen = min(cands, key=lambda x: (x["array_estimate"], x["cols"]))
            r["cols"] = chosen["cols"]
            r["group"] = g
            r["panel_bytes"] = chosen["panel_bytes"]
            r["candidates"] = cands
            r["cross_group_carry"] = bool(chosen["cols"] % 32 and r["n"] > chosen["cols"])
    # Fused linear->GELU->linear chains: carry sizing and output format follow
    # group_k; recomputed from the (possibly re-tiled) producer matrices.
    for cmd in p["commands"]:
        f = p.get("group_stream", {}).get(str(cmd["id"]))
        if f and cmd["matrices"]:
            r = cmd["matrices"][0]
            f["output_format"] = f"row-major K{g} INT8 + 8-byte scale record"
            f["carry_bytes"] = r["batch"] * r["m"] * g
            if r["cols"] % 32 and r["n"] > r["cols"]:
                f["cross_group_carry"] = True
                carry_chains += 1
    p["formats"]["static_gemm_count"] = static_cnt
    p["formats"]["dynamic_gemm_count"] = dyn_cnt
    p["formats"]["fused_chains_with_cross_group_carry"] = carry_chains
    return p
