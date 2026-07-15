from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def draw_box(
    ax,
    xy,
    width,
    height,
    title,
    lines,
    facecolor,
    edgecolor="#1f2937",
    linewidth=1.35,
    rounding_size=0.035,
    shadow=False,
):
    x, y = xy
    if shadow:
        shadow_box = FancyBboxPatch(
            (x + 0.035, y - 0.035),
            width,
            height,
            boxstyle=f"round,pad=0.025,rounding_size={rounding_size}",
            linewidth=0,
            edgecolor="none",
            facecolor="#111827",
            alpha=0.10,
            zorder=1,
        )
        ax.add_patch(shadow_box)

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.025,rounding_size={rounding_size}",
        linewidth=linewidth,
        edgecolor=edgecolor if linewidth > 0 else "none",
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height - 0.18,
        title,
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#111827",
        zorder=3,
    )
    body_fontsize = 7.7 if len(lines) >= 5 else 9.0
    ax.text(
        x + width / 2,
        y + height - 0.42,
        "\n".join(lines),
        ha="center",
        va="top",
        fontsize=body_fontsize,
        linespacing=1.1,
        color="#111827",
        zorder=3,
    )


def draw_arrow(ax, start, end, text=None, text_offset=(0, 0)):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="Simple,head_length=10,head_width=10,tail_width=2.4",
        mutation_scale=1,
        linewidth=0,
        color="#374151",
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2 + text_offset[0]
        mid_y = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(
            mid_x,
            mid_y,
            text,
            ha="center",
            va="center",
            fontsize=8.8,
            color="#374151",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#ffffff", "edgecolor": "none"},
        )


def build_diagram(output_stem: str = "system_workflow_diagram") -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16.5, 6.2))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        8.25,
        6.25,
        "Speech Command Recognition Workflow for Robot Navigation",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        8.25,
        5.88,
        "Audio input -> preprocessing -> Log-Mel feature extraction -> CNN classifier -> command label -> robot action",
        ha="center",
        va="center",
        fontsize=11,
        color="#4b5563",
    )

    width = 1.95
    height = 1.42
    y = 3.55
    gap = 0.70
    boxes = [
        (
            "Audio Input",
            ["WAV file", "or microphone", "1 short command"],
            "#dbeafe",
        ),
        (
            "Preprocessing",
            ["mono audio", "16 kHz resample", "1 s trim/pad", "amplitude normalize"],
            "#dcfce7",
        ),
        (
            "Feature Extraction",
            ["Log-Mel", "spectrogram", "64 Mel bins"],
            "#fef3c7",
        ),
        (
            "CNN Model",
            ["Conv blocks", "BatchNorm + ReLU", "softmax scores"],
            "#fce7f3",
        ),
        (
            "Decision",
            ["choose top label", "check confidence", "low score -> unknown"],
            "#ede9fe",
        ),
        (
            "Robot Action",
            ["forward", "backward", "left", "right", "stop", "unknown"],
            "#e0f2fe",
        ),
    ]

    x_positions = [0.5 + i * (width + gap) for i in range(len(boxes))]
    for x, (title, lines, color) in zip(x_positions, boxes):
        draw_box(
            ax,
            (x, y),
            width,
            height,
            title,
            lines,
            color,
            edgecolor="none",
            linewidth=0,
            rounding_size=0.09,
            shadow=True,
        )

    for i in range(len(x_positions) - 1):
        start = (x_positions[i] + width, y + height / 2)
        end = (x_positions[i + 1], y + height / 2)
        draw_arrow(ax, start, end)

    lower_y = 1.05
    train_x = x_positions[1] + 0.15
    eval_x = x_positions[3] + 0.1
    draw_box(
        ax,
        (train_x, lower_y),
        2.7,
        1.35,
        "Training Data",
        ["Google Speech Commands v2", "target navigation words", "other words -> unknown"],
        "#f3f4f6",
        edgecolor="#1f2937",
        linewidth=1.35,
        rounding_size=0.045,
    )
    draw_box(
        ax,
        (eval_x, lower_y),
        2.7,
        1.35,
        "Evaluation Outputs",
        ["accuracy and macro F1", "classification report", "confusion matrix"],
        "#f3f4f6",
        edgecolor="#1f2937",
        linewidth=1.35,
        rounding_size=0.045,
    )

    draw_arrow(
        ax,
        (train_x + 2.7, lower_y + 1.02),
        (x_positions[3] + 0.25, y),
        "model training",
        text_offset=(0.1, -0.12),
    )
    draw_arrow(
        ax,
        (x_positions[3] + width * 0.5, y),
        (eval_x + 1.35, lower_y + 1.35),
        "validation/test",
        text_offset=(0.0, 0.08),
    )

    png_path = FIGURES_DIR / f"{output_stem}.png"
    pdf_path = FIGURES_DIR / f"{output_stem}.pdf"
    svg_path = FIGURES_DIR / f"{output_stem}.svg"
    fig.savefig(png_path, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png_path, pdf_path, svg_path]


def main() -> None:
    for output_path in build_diagram():
        print(output_path)


if __name__ == "__main__":
    main()
