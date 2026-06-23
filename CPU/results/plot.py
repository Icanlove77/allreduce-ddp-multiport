import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


CSV_PATH = Path("9nodes_5times_cpu.csv")

# All algorithms now run with the same world_size=9, so no normalization is needed.
METRIC = "avg_step_time_ms"

OUT_AVG_FIG = Path("9nodes_5times_cpu_relative_completion_mean.pdf")
OUT_FAV_FIG = Path("9nodes_5times_cpu_relative_completion_best.pdf")

families = {
    "BuiltIn": ["builtin"],
    "Ring": ["ring"],
    "RecDoub": [
        "recursive-doubling-latency",
        "recursive-doubling-bandwidth",
    ],
    "Swing": [
        "swing-latency",
        "swing-bandwidth",
    ],
    "Bruck": [
        "bruck-latency",
        "bruck-bandwidth",
    ],
    "Trivance": [
        "trivance-latency",
        "trivance-bandwidth",
    ],
}


def format_size(num_bytes: int) -> str:
    """Format estimated allreduce size."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        val = num_bytes / 1024
        return f"{val:.0f} KiB" if val >= 10 else f"{val:.1f} KiB"
    val = num_bytes / (1024 ** 2)
    return f"{val:.2f} MiB" if val < 10 else f"{val:.1f} MiB"


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)

    if METRIC not in df.columns:
        raise ValueError(
            f"Metric column '{METRIC}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # Sort model configurations by estimated allreduce size.
    meta = (
        df.drop_duplicates("model_label")
        .set_index("model_label")
        .sort_values("grad_size_bytes")
    )

    model_order = meta.index.tolist()
    x_labels = [
        format_size(int(meta.loc[m, "grad_size_bytes"]))
        for m in model_order
    ]

    return df, meta, model_order, x_labels


def build_average_summary(df: pd.DataFrame, meta: pd.DataFrame, model_order):
    """
    Plot 1:
    For each model size and algorithm family:
    - take average for latency/bandwidth variants;
    - choose the better one;
    - compare against Trivance's better average.
    """
    rows = []

    for model_label in model_order:
        for family, algos in families.items():
            sub = df[
                (df["model_label"] == model_label)
                & (df["algo"].isin(algos))
            ]

            if sub.empty:
                continue

            if family in ["BuiltIn", "Ring"]:
                selected_algo = algos[0]
                selected_time = sub[METRIC].mean()
            else:
                means = sub.groupby("algo")[METRIC].mean()
                selected_algo = means.idxmin()
                selected_time = means.min()

            rows.append(
                {
                    "model_label": model_label,
                    "family": family,
                    "selected_algo": selected_algo,
                    "selected_time_ms": float(selected_time),
                    "grad_size_bytes": int(meta.loc[model_label, "grad_size_bytes"]),
                    "grad_size_mib": float(meta.loc[model_label, "grad_size_mib"]),
                }
            )

    summary = pd.DataFrame(rows)

    trivance_base = (
        summary[summary["family"] == "Trivance"][
            ["model_label", "selected_time_ms"]
        ]
        .rename(columns={"selected_time_ms": "trivance_time_ms"})
    )

    summary = summary.merge(trivance_base, on="model_label", how="left")
    summary["relative_vs_trivance_pct"] = (
        (summary["selected_time_ms"] / summary["trivance_time_ms"] - 1.0)
        * 100.0
    )

    return summary


def build_trivance_favored_summary(df: pd.DataFrame, meta: pd.DataFrame, model_order):
    """
    Plot 2:
    - Trivance takes the best result among runs.
    - For latency/bandwidth algorithms, still choose the better variant within each run.
    """
    rows = []

    for model_label in model_order:
        for family, algos in families.items():
            sub = df[
                (df["model_label"] == model_label)
                & (df["algo"].isin(algos))
            ]

            if sub.empty:
                continue

            if family == "Trivance":
                best_by_repeat = sub.groupby("repeat_id")[METRIC].min()
                selected_time = best_by_repeat.min()
                selected_algo = "trivance_best_repeat"

            elif family in ["BuiltIn", "Ring"]:
                selected_time = sub.groupby("repeat_id")[METRIC].max().max()
                selected_algo = f"{family.lower()}_worst_repeat"

            else:
                best_by_repeat = sub.groupby("repeat_id")[METRIC].min()
                selected_time = best_by_repeat.max()
                selected_algo = f"{family.lower()}_best_variant_worst_repeat"

            rows.append(
                {
                    "model_label": model_label,
                    "family": family,
                    "selected_algo": selected_algo,
                    "selected_time_ms": float(selected_time),
                    "grad_size_bytes": int(meta.loc[model_label, "grad_size_bytes"]),
                    "grad_size_mib": float(meta.loc[model_label, "grad_size_mib"]),
                }
            )

    summary = pd.DataFrame(rows)

    trivance_base = (
        summary[summary["family"] == "Trivance"][
            ["model_label", "selected_time_ms"]
        ]
        .rename(columns={"selected_time_ms": "trivance_time_ms"})
    )

    summary = summary.merge(trivance_base, on="model_label", how="left")
    summary["relative_vs_trivance_pct"] = (
        (summary["selected_time_ms"] / summary["trivance_time_ms"] - 1.0)
        * 100.0
    )

    return summary


def find_transition_points(df: pd.DataFrame, model_order):
    """
    Find first model size where bandwidth version beats latency version.
    """
    pairs = {
        "RecDoub": (
            "recursive-doubling-latency",
            "recursive-doubling-bandwidth",
        ),
        "Swing": (
            "swing-latency",
            "swing-bandwidth",
        ),
        "Bruck": (
            "bruck-latency",
            "bruck-bandwidth",
        ),
        "Trivance": (
            "trivance-latency",
            "trivance-bandwidth",
        ),
    }

    transition_points = {}

    for family, (latency_algo, bandwidth_algo) in pairs.items():
        transition_idx = None

        for i, model_label in enumerate(model_order):
            sub = df[
                (df["model_label"] == model_label)
                & (df["algo"].isin([latency_algo, bandwidth_algo]))
            ]

            if sub.empty:
                continue

            means = sub.groupby("algo")[METRIC].mean()

            latency_time = means.get(latency_algo, np.inf)
            bandwidth_time = means.get(bandwidth_algo, np.inf)

            if bandwidth_time < latency_time:
                transition_idx = i
                break

        transition_points[family] = transition_idx

    return transition_points


def add_trivance_transition_line(ax, transition_points, x_labels):
    """
    Draw a vertical dashed line for Trivance's first transition point:
    the first model size where Trivance bandwidth beats Trivance latency.
    """
    idx = transition_points.get("Trivance")

    if idx is None:
        return

    ax.axvline(
        x=idx,
        linestyle=":",
        linewidth=2.5,
        color="0.35",
        alpha=0.9,
        zorder=0,
    )


def add_trivance_direction_annotation(ax):
    """
    Add right-side green/red arrows:
    upper arrow: Trivance better
    lower arrow: Trivance worse
    """
    ymin, ymax = ax.get_ylim()

    if ymax == ymin:
        zero_frac = 0.5
    else:
        zero_frac = (0.0 - ymin) / (ymax - ymin)

    zero_frac = float(np.clip(zero_frac, 0.18, 0.82))

    x_arrow = 1.035
    x_text = 1.070
    gap = 0.018

    green = "green"
    red = "red"

    ax.annotate(
        "",
        xy=(x_arrow, 0.97),
        xytext=(x_arrow, zero_frac + gap),
        xycoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=5.5,
            color=green,
            mutation_scale=34,
            shrinkA=0,
            shrinkB=0,
        ),
        annotation_clip=False,
    )

    ax.annotate(
        "",
        xy=(x_arrow, 0.03),
        xytext=(x_arrow, zero_frac - gap),
        xycoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=5.5,
            color=red,
            mutation_scale=34,
            shrinkA=0,
            shrinkB=0,
        ),
        annotation_clip=False,
    )

    ax.text(
        x_text,
        (zero_frac + 0.97) / 2,
        "TRIVANCE\nBETTER",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=18,
        color=green,
        fontweight="bold",
        linespacing=1.15,
        clip_on=False,
    )

    ax.text(
        x_text,
        (0.03 + zero_frac) / 2,
        "TRIVANCE\nWORSE",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=18,
        color=red,
        fontweight="bold",
        linespacing=1.15,
        clip_on=False,
    )


def plot_summary(summary: pd.DataFrame, model_order, x_labels, transition_points, out_path: Path, title: str):
    x = np.arange(len(model_order))

    fig, ax = plt.subplots(figsize=(11.2, 6.5))

    plot_families = ["BuiltIn", "Ring", "RecDoub", "Swing", "Bruck"]

    styles = {
        "BuiltIn": {
            "marker": "D",
            "color": "#ff7f0e",
        },
        "Ring": {
            "marker": "o",
            "color": "#1f77b4",
        },
        "RecDoub": {
            "marker": "s",
            "color": "#2ca02c",
        },
        "Swing": {
            "marker": "^",
            "color": "#d62728",
        },
        "Bruck": {
            "marker": "v",
            "color": "#9467bd",
        },
    }

    for family in plot_families:
        sub = (
            summary[summary["family"] == family]
            .set_index("model_label")
            .loc[model_order]
        )

        y = sub["relative_vs_trivance_pct"].to_numpy()

        ax.plot(
            x,
            y,
            marker=styles[family]["marker"],
            color=styles[family]["color"],
            linewidth=2.6,
            markersize=8,
            label=family,
        )

        # Hollow marker: first point where bandwidth beats latency.
        idx = transition_points.get(family)

        if idx is not None:
            ax.plot(
                x[idx],
                y[idx],
                marker=styles[family]["marker"],
                markersize=12,
                markerfacecolor="white",
                markeredgecolor=styles[family]["color"],
                markeredgewidth=2.4,
                linestyle="None",
                zorder=7,
            )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.4,
        color="0.35",
        alpha=0.85,
    )

    add_trivance_transition_line(ax, transition_points, x_labels)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=13)

    ax.set_xlabel("AllReduce Size", fontsize=22, labelpad=12)
    ax.set_ylabel("Relative Completion\nTime vs. Trivance (%)", fontsize=22, labelpad=12)

    if title:
        ax.set_title(title, fontsize=16, pad=12)

    ax.tick_params(axis="y", labelsize=14)

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.75,
        alpha=0.65,
    )

    ax.legend(
        loc="best",
        fontsize=14,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
    )

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    ax.set_ylim(ymin - 0.08 * yrange, ymax + 0.08 * yrange)

    add_trivance_direction_annotation(ax)

    fig.subplots_adjust(right=0.78, bottom=0.23)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df, meta, model_order, x_labels = load_data(CSV_PATH)

    avg_summary = build_average_summary(df, meta, model_order)
    fav_summary = build_trivance_favored_summary(df, meta, model_order)

    transition_points = find_transition_points(df, model_order)

    plot_summary(
        avg_summary,
        model_order,
        x_labels,
        transition_points,
        OUT_AVG_FIG,
        "",
    )

    plot_summary(
        fav_summary,
        model_order,
        x_labels,
        transition_points,
        OUT_FAV_FIG,
        "",
    )

    print(f"Saved: {OUT_AVG_FIG}")
    print(f"Saved: {OUT_FAV_FIG}")

    print("\nTransition points based on run means:")
    for family, idx in transition_points.items():
        if idx is None:
            print(f"  {family}: no bandwidth-over-latency transition in this dataset")
        else:
            print(f"  {family}: {x_labels[idx]}")


if __name__ == "__main__":
    main()