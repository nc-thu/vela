"""Draw Fig. 12: post-route module cost and critical-path responsibility."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "fig12_data.json").read_text(encoding="utf-8"))
STEM = HERE / "fig12_impl_breakdown"

WIDTH_MM = 183.0
HEIGHT_MM = 112.0

COLORS = {
    "w8": "#287D94",
    "w4": "#DC7000",
    "group": "#C9B48C",
    "hnu": "#8298AC",
    "move": "#A6A6A6",
    "control": "#D0D0D0",
}

WORK_COLORS = {
    "Matrix products": "#287D94",
    "Group processing": "#C9B48C",
    "HNU work": "#DC7000",
    "Data movement / layout": "#8298AC",
    "Control": "#D0D0D0",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.2,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
    }
)


def fmt_value(value: float, metric: str) -> str:
    if metric in {"lut", "dsp"}:
        return f"{int(value):,}"
    if metric == "bram36":
        return f"{value:g}"
    if metric == "power_w":
        return f"{value:.3f}"
    if metric == "wns_ns":
        return f"{value:.3f}"
    raise KeyError(metric)


def critical_groups(entry: dict[str, int]) -> dict[str, int]:
    grouped = {
        "Matrix products": entry["gemm"],
        "Group processing": sum(entry[k] for k in ("quantize", "scale", "merge", "restore", "encode")),
        "HNU work": sum(entry[k] for k in ("nonlinear", "reduce", "elementwise")),
        "Data movement / layout": sum(entry[k] for k in ("ddr_write", "ddr_read", "local_move", "weight_read", "layout")),
        "Control": entry["control"],
    }
    assert sum(grouped.values()) == entry["total"], (sum(grouped.values()), entry["total"])
    return grouped


def main() -> None:
    rows = DATA["module_rows"]
    n = len(rows)
    y = np.arange(n)[::-1]
    colors = [COLORS[row["family"]] for row in rows]

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    gs = fig.add_gridspec(
        2,
        6,
        width_ratios=[2.55, 1.35, 1.12, 1.12, 1.28, 1.35],
        height_ratios=[3.25, 1.10],
        left=0.018,
        right=0.992,
        bottom=0.090,
        top=0.935,
        wspace=0.28,
        hspace=0.45,
    )

    ax_labels = fig.add_subplot(gs[0, 0])
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(-0.65, n - 0.35)
    ax_labels.axis("off")
    ax_labels.text(0.0, n - 0.05, "(a) Post-route module cost", fontsize=7.4, fontweight="bold", va="bottom")
    for yi, row in zip(y, rows, strict=True):
        ax_labels.text(0.02, yi, row["module"], va="center", ha="left", fontsize=5.65)
    ax_labels.text(0.02, n - 1.43, "alternative arrays", color="#555555", fontsize=4.9, style="italic")

    metrics = [
        ("lut", "LUT"),
        ("dsp", "DSP"),
        ("bram36", "BRAM36"),
        ("power_w", "Power (W)"),
    ]
    for col, (metric, title) in enumerate(metrics, start=1):
        ax = fig.add_subplot(gs[0, col])
        values = np.array([float(row[metric]) for row in rows])
        maximum = values.max()
        norm = values / maximum if maximum else values
        ax.barh(y, norm, height=0.60, color=colors, edgecolor="black", linewidth=0.45)
        for yi, raw, nv in zip(y, values, norm, strict=True):
            xpos = min(max(nv + 0.025, 0.035), 1.04)
            ha = "left" if xpos < 0.87 else "right"
            if ha == "right":
                xpos = 1.04
            ax.text(xpos, yi, fmt_value(raw, metric), va="center", ha=ha, fontsize=4.75)
        ax.set_xlim(0, 1.08)
        ax.set_ylim(-0.65, n - 0.35)
        ax.set_title(title, fontsize=6.0, fontweight="bold", pad=3)
        ax.set_yticks([])
        ax.set_xticks([0, 1], ["0", "max"])
        ax.tick_params(axis="x", labelsize=4.7, pad=1.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.45)
        ax.set_axisbelow(True)

    ax_w = fig.add_subplot(gs[0, 5])
    wns = np.array([float(row["wns_ns"]) for row in rows])
    ax_w.hlines(y, 0, wns, color=colors, linewidth=1.2)
    ax_w.scatter(wns, y, s=13, color=colors, edgecolor="black", linewidth=0.45, zorder=3)
    ax_w.axvline(0, color="black", linewidth=0.75)
    for yi, raw in zip(y, wns, strict=True):
        ax_w.text(raw + 0.045, yi, f"{raw:.3f}", va="center", ha="left", fontsize=4.75)
    ax_w.set_xlim(-0.05, 2.30)
    ax_w.set_ylim(-0.65, n - 0.35)
    ax_w.set_title("WNS (ns)", fontsize=6.0, fontweight="bold", pad=3)
    ax_w.set_yticks([])
    ax_w.set_xticks([0, 1, 2])
    ax_w.tick_params(axis="x", labelsize=4.7, pad=1.5)
    ax_w.spines[["top", "right", "left"]].set_visible(False)
    ax_w.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.45)
    ax_w.set_axisbelow(True)

    ax_b = fig.add_subplot(gs[1, :])
    ax_b.text(0.0, 1.25, "(b) Critical-path attribution for one invocation", transform=ax_b.transAxes,
              fontsize=7.4, fontweight="bold", va="bottom")
    configs = ["W8A8 G64", "W4A8 G32"]
    ypos = [1, 0]
    left = np.zeros(2)
    grouped_by_config = {name: critical_groups(DATA["critical_cycles"][name]) for name in configs}
    for work, color in WORK_COLORS.items():
        widths = np.array([
            grouped_by_config[name][work] / DATA["critical_cycles"][name]["total"] * 100
            for name in configs
        ])
        ax_b.barh(ypos, widths, left=left, height=0.50, color=color, edgecolor="black", linewidth=0.55, label=work)
        for yy, ll, ww in zip(ypos, left, widths, strict=True):
            if ww >= 4.0:
                ax_b.text(ll + ww / 2, yy, f"{ww:.1f}%", ha="center", va="center", fontsize=5.1,
                          color="white" if work in {"Matrix products", "HNU work", "Data movement / layout"} else "black")
        left += widths

    for yy, name in zip(ypos, configs, strict=True):
        total = DATA["critical_cycles"][name]["total"]
        latency_ms = total / (DATA["clock_mhz"] * 1000.0)
        ax_b.text(-0.8, yy, name, va="center", ha="right", fontsize=5.7)
        ax_b.text(100.7, yy, f"{latency_ms:.2f} ms", va="center", ha="left", fontsize=5.8, fontweight="bold")

    ax_b.set_xlim(-10, 112)
    ax_b.set_ylim(-0.62, 1.62)
    ax_b.set_yticks([])
    ax_b.set_xlabel("Share of modeled critical-path cycles (%)", fontsize=5.9, labelpad=2)
    ax_b.set_xticks([0, 20, 40, 60, 80, 100])
    ax_b.tick_params(axis="both", labelsize=5.4, pad=1.5)
    ax_b.spines[["top", "right", "left"]].set_visible(False)
    ax_b.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.45)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="lower right", bbox_to_anchor=(1.0, 1.13), ncol=5, fontsize=5.2,
                handlelength=1.5, columnspacing=1.4, handletextpad=0.4)

    fig.text(
        0.5,
        0.012,
        "Module-level vectorless power estimates; cycle shares follow critical-path attribution.",
        ha="center",
        va="bottom",
        fontsize=5.25,
        color="#333333",
    )

    fig.savefig(STEM.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(STEM.with_suffix(".svg"), dpi=600, facecolor="white")
    plt.close(fig)

    derived = {
        "clock_mhz": DATA["clock_mhz"],
        "critical_path": {
            name: {
                "total_cycles": DATA["critical_cycles"][name]["total"],
                "latency_ms": DATA["critical_cycles"][name]["total"] / (DATA["clock_mhz"] * 1000.0),
                "groups_cycles": grouped_by_config[name],
                "groups_percent": {
                    k: v / DATA["critical_cycles"][name]["total"] * 100.0
                    for k, v in grouped_by_config[name].items()
                },
            }
            for name in configs
        },
    }
    (HERE / "fig12_derived.json").write_text(json.dumps(derived, indent=2), encoding="utf-8")
    print(STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
