"""Finite local G32 -> FP16 GELU -> K64 stream. No numerical quality claim."""
import math

C = lambda n, d: (n + d - 1) // d


class StreamMixin:
    def quantize(self, h, rows, k, base, side):
        f = self.formats.get(h["key"], {})
        if f.get("kind") == "g64_after_gelu":
            assert side == "a" and math.prod(f["shape"][:-1]) == rows and f["shape"][-1] == k
            assert h.get("offset", 0) == 0
            self.stats["input_quantization_bypassed_values"] += rows * k
            return {
                **h,
                "bytes": rows * C(k, self.fmt_ga) * self.fmt_gb,
                "width": 1,
                "compact": False,
                "borrowed": True,
            }, base
        return super().quantize(h, rows, k, base, side)

    def prepare_g64_scales(self, qa, r, base):
        groups = C(r["k"], self.fmt_ga)
        rows = r["batch"] * r["m"]
        # Read actual G64 scale records. Weights are constant: per-K-group maxima
        # over output channels can be precomputed offline, not zero-byte metadata.
        base = self.grouped_read(
            qa["key"], rows * groups, 8, base, self.fmt_gb - 8, self.fmt_gb, 64
        )
        base = self.dma(
            "weight_scale_max:" + r["weight_storage"],
            C(r["k"], self.fmt_gw) * 8,
            base,
            False,
            "weight",
            "offline per-K-group weight-scale maxima for common exponent",
            64,
        )
        base = self.emit(
            "scale",
            C(rows * groups, 32) + groups + 16,
            base,
            resource="scale",
            reason="reduce activation scales by K group; multiply offline weight maxima; common Q24 exponent",
        )
        self.stats["ready_g64_scale_records_read"] += rows * groups
        return base

    def finish_temp(self, h, base):
        if h.get("borrowed"):
            return
        return super().finish_temp(h, base)

    def start_stream(self, f, b, m, n, base):
        self.stream_context = {"spec": f, "slots": [base, base], "tile": 0, "n": n}
        carry, base = self.temp(b * m * self.fmt_ga, base, "group_carry")
        self.stream_context["carry"] = carry
        if f["bias"] is not None:
            h = self.handle(f["bias"])
            base = self.data_read(h, h["bytes"], base)
        self.stream_context["ready"] = base
        self.stats["fused_linear_gelu_chains"] += 1
        self.stats["carry_bytes_peak"] = max(self.stats["carry_bytes_peak"], b * m * self.fmt_ga)
        return base

    def stream_tile(self, key, bi, row, mr, col, nc, m, n, base):
        c = self.stream_context
        slot = c["tile"] % 2
        c["tile"] += 1
        start = self.end(base, c["slots"][slot])
        birth = self.s.end(start)
        # One <=16x96 G32 tile plus its FP16 values fits the existing 32 KiB slot.
        size = mr * C(nc, 32) * 40 + mr * nc * 2 + mr * self.fmt_ga * 2
        assert size <= 32768
        x = self.emit(
            "restore",
            C(mr * nc, 32) + 4,
            start,
            resource="stream_dequant",
            reason="G32 INT8 and local scale -> FP16; no whole-tensor normalization",
        )
        if c["spec"]["bias"] is not None:
            x = self.emit(
                "elementwise",
                C(mr * nc, 32) + 8,
                x,
                resource="vector",
                reason="FP16 bias before GELU; resident bias reused",
            )
        x = self.elementwise_sfu("gelu", mr * nc, x)
        # 96-wide output panels leave 32 values per row for the next panel.
        if col % self.fmt_ga:
            x = self.grouped_read(
                c["carry"],
                mr,
                (col % self.fmt_ga) * 2,
                x,
                (bi * m + row) * self.fmt_ga,
                self.fmt_ga,
                64,
            )
            self.stats["carry_read_values"] += mr * (col % self.fmt_ga)
        completed = (col + nc) // self.fmt_ga - col // self.fmt_ga
        tail = col + nc == n and n % self.fmt_ga != 0
        completed += int(tail)
        values = completed * self.fmt_ga * mr
        if completed:
            x = self.emit(
                "quantize",
                2 * C(values, 32) + completed * mr + 2,
                self.end(x, self.quant_done),
                resource="quant",
                reason="post-GELU local group absmax + RNE INT8; bounded group scratch",
            )
            self.quant_done = x
            self.stats["post_gelu_g64_groups"] += mr * completed
            for rr in range(mr):
                off = ((bi * m + row + rr) * C(n, self.fmt_ga) + col // self.fmt_ga) * self.fmt_gb
                x = self.store(key, completed * self.fmt_gb, x, "group_output", off)
        if (col + nc) % self.fmt_ga and not tail:
            x = self.store(
                c["carry"],
                mr * ((col + nc) % self.fmt_ga) * 2,
                x,
                "group_carry",
                (bi * m + row) * self.fmt_ga,
            )
            self.stats["carry_write_values"] += mr * ((col + nc) % self.fmt_ga)
        c["slots"][slot] = x
        self.streams.append(
            {"op": self.op, "slot": slot, "start": birth, "end": self.s.end(x), "bytes": size}
        )
        return x

    def finish_stream(self, base):
        self.finish_temp({"key": self.stream_context["carry"]}, base)

    def forward_gelu(self, h, key, base):
        # The producer already ran bias/GELU and generated K64 records.
        old = h["key"]
        assert old != key
        self.objects[key]["bytes"] = self.objects[old]["bytes"]
        self.formats[key] = dict(self.formats[old])
        if old in self.cache:
            self.cache[key] = self.cache.pop(old)
            self.hazards[key] = self.hazards.pop(old, {})
            self.last_access[key] = self.last_access.get(old, base)
            self.allocations.append(
                {"op": self.op, "action": "rename", "key": old, "new_key": key, "after": base}
            )
        elif old in self.backing:
            self.backing.add(key)
        self.pinned.discard(old)
        self.stats["elided_standalone_gelu"] += 1
        return {**h, "key": key}, base
