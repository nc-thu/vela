import collections
from .previous_schedule import build as previous


def build(*args, **kwargs):
    p = previous(*args, **kwargs)
    p["version"] = 6
    users = collections.defaultdict(list)
    for c in p["commands"]:
        for t in c["inputs"]:
            users[t].append(c)
    fused = {}
    for c in p["commands"]:
        if c["op"].split(".")[1] != "linear" or len(c["matrices"]) != 1 or len(c["outputs"]) != 1:
            continue
        r = c["matrices"][0]
        us = users[c["outputs"][0]]
        if len(us) != 1 or us[0]["op"].split(".")[1] != "gelu":
            continue
        g = us[0]
        if len(g["outputs"]) != 1:
            continue
        nxt = users[g["outputs"][0]]
        if len(nxt) != 1 or nxt[0]["op"].split(".")[1] != "linear":
            continue
        d = nxt[0]
        if d["inputs"][0] != g["outputs"][0] or len(d["matrices"]) != 1:
            continue
        if r["action"] or r["cols"] % 32 or r["n"] % 32 or not r["constant_b"]:
            continue
        if d["matrices"][0]["k"] != r["n"]:
            continue
        # Require no live storage aliases of the elided intermediate tensors.
        mids = c["outputs"] + g["outputs"]
        keys = [p["tensor_storage"][str(t)]["storage"] for t in mids]
        if any(sum(v["storage"] == k for v in p["tensor_storage"].values()) != 1 for k in keys):
            continue
        fused[str(c["id"])] = {
            "linear": c["id"],
            "gelu": g["id"],
            "consumer": d["id"],
            "bias": c["inputs"][2] if len(c["inputs"]) > 2 else None,
            "output_format": "row-major K64 INT8 + 8-byte scale record",
            "encoding": "G32 local output -> FP16 dequant/bias/GELU -> G64 RNE",
            "carry_bytes": r["batch"] * r["m"] * 32 * 2,
        }
    p["group_stream"] = fused
    p["execution_contract"][
        "quality"
    ] = "Grouped FP16 interface; no v12 integer-global-normalization equivalence claim; quality untested"
    p["execution_contract"][
        "streaming"
    ] = "Eligible exclusive linear-GELU-linear chains only; same matrix array and 32-lane SFU; finite two scratch slots; K64 carry stored in arena"
    return p
