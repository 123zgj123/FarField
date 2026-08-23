"""Render the README hero figure.

Not part of the runtime. Needs matplotlib (optional, not a FarField dependency):

    python3 assets/render_overview.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

OUT = Path(__file__).with_name("overview.png")

BG = "#f7f8fb"
INK = "#1e2a3a"
MUTED = "#667084"
ARROW = "#b7c0cc"
SERIF = "Linux Libertine O"
SANS = "Noto Sans"


def chevron(ax, x, y, color, scale=0.11, z=6):
    ax.add_patch(
        Polygon(
            [(x - scale, y - scale * 0.85), (x + scale * 0.42, y), (x - scale, y + scale * 0.85)],
            closed=True,
            fc=color,
            ec=color,
            lw=0,
            zorder=z,
        )
    )


def icon_topic(ax, cx, cy, color, s=1.0):
    ax.add_patch(Circle((cx, cy), 0.14 * s, fill=False, lw=1.8, ec=color, zorder=5))
    ax.text(cx, cy - 0.012, "?", ha="center", va="center", fontsize=11 * s, color=color, fontweight="bold", fontfamily=SANS, zorder=5)


def icon_search(ax, cx, cy, color, s=1.0):
    d = 0.13 * s
    pts = [(cx, cy + d), (cx - d, cy - 0.08 * s), (cx + d, cy - 0.08 * s)]
    ax.add_patch(Polygon(pts, closed=True, fill=False, lw=1.7, ec=color, zorder=5))
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.036 * s, fc=color, ec=color, zorder=6))


def icon_hyp(ax, cx, cy, color, s=1.0):
    ax.add_patch(Rectangle((cx - 0.11 * s, cy - 0.13 * s), 0.22 * s, 0.26 * s, fill=False, lw=1.7, ec=color, zorder=5))
    for dy in (-0.045, 0.02, 0.085):
        ax.plot([cx - 0.065 * s, cx + 0.065 * s], [cy + dy * s, cy + dy * s], color=color, lw=1.4, zorder=5, solid_capstyle="round")


def icon_screen(ax, cx, cy, color, s=1.0):
    ax.add_patch(
        Polygon(
            [
                (cx - 0.14 * s, cy + 0.12 * s),
                (cx + 0.14 * s, cy + 0.12 * s),
                (cx + 0.052 * s, cy - 0.02 * s),
                (cx - 0.052 * s, cy - 0.02 * s),
            ],
            closed=True,
            fill=False,
            lw=1.7,
            ec=color,
            zorder=5,
        )
    )
    ax.plot([cx, cx], [cy - 0.02 * s, cy - 0.13 * s], color=color, lw=1.7, zorder=5, solid_capstyle="round")


def icon_exp(ax, cx, cy, color, s=1.0):
    ax.add_patch(Rectangle((cx - 0.115 * s, cy - 0.13 * s), 0.08 * s, 0.17 * s, fc=color, ec=color, lw=0, zorder=5, alpha=0.92))
    ax.add_patch(Rectangle((cx + 0.035 * s, cy - 0.13 * s), 0.08 * s, 0.26 * s, fill=False, lw=1.7, ec=color, zorder=5))


def icon_proto(ax, cx, cy, color, s=1.0):
    ax.add_patch(Rectangle((cx - 0.07 * s, cy - 0.145 * s), 0.19 * s, 0.23 * s, fill=False, lw=1.7, ec=color, zorder=5))
    ax.add_patch(Rectangle((cx - 0.13 * s, cy - 0.095 * s), 0.19 * s, 0.23 * s, fill=False, lw=1.7, ec=color, zorder=5))


def main() -> None:
    plt.rcParams["font.family"] = SANS
    W, H = 12.6, 4.2
    fig, ax = plt.subplots(figsize=(W, H), dpi=240)
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0.42, H - 0.32, "FarField", color=INK, fontsize=20, fontfamily=SERIF, fontstyle="italic", va="center", zorder=4)

    stages = [
        ("Topic", "from the user", icon_topic, "#4c8bf5"),
        ("Search", "papers + graph", icon_search, "#2bbbad"),
        ("Hypothesis", "claim + prediction", icon_hyp, "#8b6ce0"),
        ("Screening", "novelty filters", icon_screen, "#e09a2b"),
        ("Experiment", "two arms, frozen data", icon_exp, "#ee6c4d"),
        ("Protocol", "packet + protocol.json", icon_proto, "#4f6bd6"),
    ]

    n = len(stages)
    left, right = 0.95, W - 0.95
    xs = [left + i * (right - left) / (n - 1) for i in range(n)]
    cy_num = 2.48
    r = 0.46
    icon_s = 1.55

    for i in range(n - 1):
        ax.plot(
            [xs[i] + r + 0.14, xs[i + 1] - r - 0.22],
            [cy_num, cy_num],
            color=ARROW,
            lw=1.8,
            zorder=2,
            solid_capstyle="round",
        )
        chevron(ax, xs[i + 1] - r - 0.12, cy_num, ARROW, scale=0.10)

    for i, ((title, body, icon, accent), cx) in enumerate(zip(stages, xs), start=1):
        ax.add_patch(Circle((cx, cy_num), r + 0.18, fc=accent, ec="none", alpha=0.12, zorder=1))
        ax.add_patch(Circle((cx, cy_num), r, fc=accent, ec="none", zorder=3))
        ax.text(cx, cy_num - 0.02, str(i), ha="center", va="center", fontsize=18, color="#ffffff", fontfamily=SERIF, zorder=4)
        icon(ax, cx, cy_num - 0.92, accent, s=icon_s)
        ax.text(cx, cy_num - 1.38, title, ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK, fontfamily=SANS, zorder=4)
        ax.text(cx, cy_num - 1.68, body, ha="center", va="center", fontsize=8.6, color=MUTED, fontfamily=SANS, zorder=4)

    fig.savefig(OUT, dpi=240, facecolor=BG, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
