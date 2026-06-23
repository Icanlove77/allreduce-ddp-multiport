import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "small_vector_size_3gpu_results.csv"

METRIC = "avg_step_time_ms"
OUT_DIR = SCRIPT_DIR / "small_vector_plots_trivance_best"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ABS_PDF = OUT_DIR / "small_vector_3gpu_absolute_avg_step_time_trivance_best.pdf"
OUT_ABS_PNG = OUT_DIR / "small_vector_3gpu_absolute_avg_step_time_trivance_best.png"
OUT_REL_PDF = OUT_DIR / "small_vector_3gpu_relative_vs_trivance_best_mean.pdf"
OUT_REL_PNG = OUT_DIR / "small_vector_3gpu_relative_vs_trivance_best_mean.png"
OUT_FAV_PDF = OUT_DIR / "small_vector_3gpu_relative_vs_trivance_best_favored.pdf"
OUT_FAV_PNG = OUT_DIR / "small_vector_3gpu_relative_vs_trivance_best_favored.png"
OUT_SUMMARY = OUT_DIR / "small_vector_3gpu_trivance_best_summary.csv"
OUT_REL_SUMMARY = OUT_DIR / "small_vector_3gpu_relative_vs_trivance_best_mean_summary.csv"
OUT_FAV_SUMMARY = OUT_DIR / "small_vector_3gpu_relative_vs_trivance_best_favored_summary.csv"

families = {
    "BuiltIn": ["builtin"],
    "Ring": ["ring"],
    "Trivance": ["trivance-latency", "trivance-bandwidth"],
}

PLOT_ORDER_ABS = ["BuiltIn", "Ring", "Trivance"]
PLOT_ORDER_REL = ["BuiltIn", "Ring"]  

styles = {
    "BuiltIn": {"marker": "D", "color": "#ff7f0e"},
    "Ring": {"marker": "o", "color": "#1f77b4"},
    "Trivance": {"marker": "^", "color": "#2ca02c"},
}


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        val = num_bytes / 1024
        return f"{val:.0f} KiB" if val >= 10 else f"{val:.1f} KiB"
    val = num_bytes / (1024 ** 2)
    return f"{val:.0f} MiB" if val >= 10 else f"{val:.1f} MiB"


def load_data():
    df = pd.read_csv(CSV_PATH)
    require_columns(
        df,
        [
            "algo",
            "size_label",
            "actual_vector_size_bytes",
            "target_vector_size_bytes",
            METRIC,
            "repeat_id",
        ],
    )

    # Keep successful rows only if the column exists.
    if "accepted" in df.columns:
        # accepted may be bool or string depending on how pandas reads it.
        df = df[df["accepted"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    for col in [METRIC, "actual_vector_size_bytes", "target_vector_size_bytes", "repeat_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[METRIC, "actual_vector_size_bytes", "target_vector_size_bytes"])

    # Sort by target size, but use actual_vector_size_bytes for the x-axis label.
    meta = (
        df.drop_duplicates("size_label")
        .set_index("size_label")
        .sort_values("target_vector_size_bytes")
    )
    size_order = meta.index.tolist()
    x_labels = [format_size(int(meta.loc[s, "actual_vector_size_bytes"])) for s in size_order]
    return df, meta, size_order, x_labels


def print_available_algorithm_report(df):
    available = set(df["algo"].dropna().unique())
    print("\nAvailable algorithms in this CSV:")
    for family, algos in families.items():
        present = [a for a in algos if a in available]
        print(f"  {family}: {present if present else 'missing'}")


def build_mean_summary(df, meta, size_order):
    rows = []
    for size_label in size_order:
        for family, algos in families.items():
            sub = df[(df["size_label"] == size_label) & (df["algo"].isin(algos))]
            if sub.empty:
                continue
            means = sub.groupby("algo")[METRIC].mean()
            selected_algo = means.idxmin()
            selected_time = float(means.min())
            rows.append({
                "size_label": size_label,
                "family": family,
                "selected_algo": selected_algo,
                "selected_time_ms": selected_time,
                "actual_vector_size_bytes": int(meta.loc[size_label, "actual_vector_size_bytes"]),
                "target_vector_size_bytes": int(meta.loc[size_label, "target_vector_size_bytes"]),
            })
    summary = pd.DataFrame(rows)
    return add_relative_to_trivance(summary)


def build_favored_summary(df, meta, size_order):
    rows = []
    for size_label in size_order:
        for family, algos in families.items():
            sub = df[(df["size_label"] == size_label) & (df["algo"].isin(algos))]
            if sub.empty:
                continue
            if family == "Trivance":
                idx = sub[METRIC].idxmin()
                selected_algo = str(sub.loc[idx, "algo"])
                selected_time = float(sub.loc[idx, METRIC])
            else:
                # For single-algo families, worst repeat.
                selected_algo = str(sub["algo"].iloc[0]) + "_worst_repeat"
                selected_time = float(sub.groupby("repeat_id")[METRIC].max().max())
            rows.append({
                "size_label": size_label,
                "family": family,
                "selected_algo": selected_algo,
                "selected_time_ms": selected_time,
                "actual_vector_size_bytes": int(meta.loc[size_label, "actual_vector_size_bytes"]),
                "target_vector_size_bytes": int(meta.loc[size_label, "target_vector_size_bytes"]),
            })
    summary = pd.DataFrame(rows)
    return add_relative_to_trivance(summary)


def add_relative_to_trivance(summary):
    base = (
        summary[summary["family"] == "Trivance"]
        [["size_label", "selected_time_ms", "selected_algo"]]
        .rename(columns={"selected_time_ms": "trivance_time_ms", "selected_algo": "trivance_selected_algo"})
    )
    if base.empty:
        raise ValueError("No Trivance rows found. Need trivance-latency and/or trivance-bandwidth.")
    out = summary.merge(base, on="size_label", how="left")
    out["relative_vs_trivance_pct"] = (out["selected_time_ms"] / out["trivance_time_ms"] - 1.0) * 100.0
    return out


def add_trivance_direction_annotation(ax):
    ymin, ymax = ax.get_ylim()
    zero_frac = 0.5 if ymax == ymin else (0.0 - ymin) / (ymax - ymin)
    zero_frac = float(np.clip(zero_frac, 0.18, 0.82))

    x_arrow = 1.035
    x_text = 1.070
    gap = 0.018

    ax.annotate(
        "",
        xy=(x_arrow, 0.97),
        xytext=(x_arrow, zero_frac + gap),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", lw=5.5, color="green", mutation_scale=34, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(x_arrow, 0.03),
        xytext=(x_arrow, zero_frac - gap),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", lw=5.5, color="red", mutation_scale=34, shrinkA=0, shrinkB=0),
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
        color="green",
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
        color="red",
        fontweight="bold",
        linespacing=1.15,
        clip_on=False,
    )


def plot_absolute(summary, size_order, x_labels):
    x_all = np.arange(len(size_order))
    size_to_x = {s: i for i, s in enumerate(size_order)}

    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    for family in PLOT_ORDER_ABS:
        sub = summary[summary["family"] == family].copy().set_index("size_label")
        present = [s for s in size_order if s in sub.index]
        if not present:
            continue
        x = np.array([size_to_x[s] for s in present])
        y = sub.loc[present, "selected_time_ms"].to_numpy()
        ax.plot(
            x,
            y,
            marker=styles[family]["marker"],
            color=styles[family]["color"],
            linewidth=2.6,
            markersize=8,
            label=family,
        )

    ax.set_xticks(x_all)
    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=13)
    ax.set_xlabel("Message Size", fontsize=22, labelpad=12)
    ax.set_ylabel("Average Step Time (ms)", fontsize=22, labelpad=12)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.65)
    ax.legend(loc="best", fontsize=14, frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.subplots_adjust(bottom=0.23)
    fig.savefig(OUT_ABS_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_ABS_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_relative(summary, size_order, x_labels, out_pdf, out_png):
    x_all = np.arange(len(size_order))
    size_to_x = {s: i for i, s in enumerate(size_order)}

    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    plotted_any = False
    for family in PLOT_ORDER_REL:
        sub = summary[(summary["family"] == family) & summary["relative_vs_trivance_pct"].notna()].copy()
        if sub.empty:
            continue
        sub = sub.set_index("size_label")
        present = [s for s in size_order if s in sub.index]
        if not present:
            continue
        x = np.array([size_to_x[s] for s in present])
        y = sub.loc[present, "relative_vs_trivance_pct"].to_numpy()
        ax.plot(
            x,
            y,
            marker=styles[family]["marker"],
            color=styles[family]["color"],
            linewidth=2.6,
            markersize=8,
            label=family,
        )
        plotted_any = True

    if not plotted_any:
        raise ValueError("No BuiltIn/Ring rows available for relative plot.")

    ax.axhline(0, linestyle="--", linewidth=1.4, color="0.35", alpha=0.85)
    ax.set_xticks(x_all)
    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=13)
    ax.set_xlabel("Message Size", fontsize=22, labelpad=12)
    ax.set_ylabel("Relative Completion\nTime vs. Trivance (%)", fontsize=22, labelpad=12)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.65)
    ax.legend(loc="best", fontsize=14, frameon=True, framealpha=0.9, edgecolor="0.8")

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    if yrange == 0:
        yrange = max(abs(ymin), 1.0)
    ax.set_ylim(ymin - 0.08 * yrange, ymax + 0.08 * yrange)
    add_trivance_direction_annotation(ax)
    fig.subplots_adjust(right=0.78, bottom=0.23)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df, meta, size_order, x_labels = load_data()
    print_available_algorithm_report(df)

    mean_summary = build_mean_summary(df, meta, size_order)
    favored_summary = build_favored_summary(df, meta, size_order)

    mean_summary.to_csv(OUT_SUMMARY, index=False)
    mean_summary.to_csv(OUT_REL_SUMMARY, index=False)
    favored_summary.to_csv(OUT_FAV_SUMMARY, index=False)

    plot_absolute(mean_summary, size_order, x_labels)
    plot_relative(mean_summary, size_order, x_labels, OUT_REL_PDF, OUT_REL_PNG)
    plot_relative(favored_summary, size_order, x_labels, OUT_FAV_PDF, OUT_FAV_PNG)

    print(f"Saved: {OUT_ABS_PDF}")
    print(f"Saved: {OUT_ABS_PNG}")
    print(f"Saved: {OUT_REL_PDF}")
    print(f"Saved: {OUT_REL_PNG}")
    print(f"Saved: {OUT_FAV_PDF}")
    print(f"Saved: {OUT_FAV_PNG}")
    print(f"Saved: {OUT_SUMMARY}")
    print("\nSelected Trivance implementation by size:")
    print(
        mean_summary[mean_summary["family"] == "Trivance"]
        [["size_label", "selected_algo", "selected_time_ms"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
