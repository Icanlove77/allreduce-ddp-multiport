import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = Path(os.environ.get("CSV_PATH", SCRIPT_DIR / "bidirectional_ring_small_vector_results.csv"))

METRIC = "avg_step_time_ms"
OUT_DIR = SCRIPT_DIR / "small_vector_plots_bidirectional_ring"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ABS_PDF = OUT_DIR / "small_vector_absolute_avg_step_time_bidirectional_ring.pdf"
OUT_ABS_PNG = OUT_DIR / "small_vector_absolute_avg_step_time_bidirectional_ring.png"
OUT_REL_PDF = OUT_DIR / "small_vector_relative_vs_bidirectional_ring_mean.pdf"
OUT_REL_PNG = OUT_DIR / "small_vector_relative_vs_bidirectional_ring_mean.png"
OUT_FAV_PDF = OUT_DIR / "small_vector_relative_vs_bidirectional_ring_favored.pdf"
OUT_FAV_PNG = OUT_DIR / "small_vector_relative_vs_bidirectional_ring_favored.png"
OUT_SUMMARY = OUT_DIR / "small_vector_bidirectional_ring_summary.csv"
OUT_REL_SUMMARY = OUT_DIR / "small_vector_relative_vs_bidirectional_ring_mean_summary.csv"
OUT_FAV_SUMMARY = OUT_DIR / "small_vector_relative_vs_bidirectional_ring_favored_summary.csv"

families = {
    "BuiltIn": ["builtin"],
    "Ring": ["ring"],
    "Bidirectional Ring": ["bidirectional-ring", "bidir-ring", "bi-ring"],
}

PLOT_ORDER_ABS = ["BuiltIn", "Ring", "Bidirectional Ring"]
PLOT_ORDER_REL = ["BuiltIn", "Ring"]

styles = {
    "BuiltIn": {"marker": "D", "color": "#ff7f0e"},
    "Ring": {"marker": "o", "color": "#1f77b4"},
    "Bidirectional Ring": {"marker": "^", "color": "#2ca02c"},
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


def normalize_columns(df):
    df = df.copy()

    if "size_label" not in df.columns:
        if "model_label" not in df.columns:
            raise ValueError("CSV must contain either size_label or model_label.")
        df["size_label"] = df["model_label"]

    if "actual_vector_size_bytes" not in df.columns:
        if "grad_size_bytes" not in df.columns:
            raise ValueError("CSV must contain either actual_vector_size_bytes or grad_size_bytes.")
        df["actual_vector_size_bytes"] = df["grad_size_bytes"]

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
    return df


def filter_successful_rows(df):
    if "accepted" not in df.columns:
        return df

    ok_values = {"true", "1", "yes", "completed"}
    return df[df["accepted"].astype(str).str.lower().isin(ok_values)].copy()


def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}. "
            "Set CSV_PATH=/path/to/bidirectional_ring_small_vector_results.csv if needed."
        )

    df = pd.read_csv(CSV_PATH)
    df = normalize_columns(df)
    df = filter_successful_rows(df)

    for col in [METRIC, "actual_vector_size_bytes", "target_vector_size_bytes", "repeat_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[METRIC, "actual_vector_size_bytes", "target_vector_size_bytes"])
    if df.empty:
        raise ValueError("No successful benchmark rows left after filtering.")

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
            rows.append(
                {
                    "size_label": size_label,
                    "family": family,
                    "selected_algo": selected_algo,
                    "selected_time_ms": selected_time,
                    "actual_vector_size_bytes": int(meta.loc[size_label, "actual_vector_size_bytes"]),
                    "target_vector_size_bytes": int(meta.loc[size_label, "target_vector_size_bytes"]),
                }
            )
    summary = pd.DataFrame(rows)
    return add_relative_to_bidirectional_ring(summary)


def build_favored_summary(df, meta, size_order):
    rows = []
    for size_label in size_order:
        for family, algos in families.items():
            sub = df[(df["size_label"] == size_label) & (df["algo"].isin(algos))]
            if sub.empty:
                continue
            if family == "Bidirectional Ring":
                idx = sub[METRIC].idxmin()
                selected_algo = str(sub.loc[idx, "algo"])
                selected_time = float(sub.loc[idx, METRIC])
            else:
                selected_algo = str(sub["algo"].iloc[0]) + "_worst_repeat"
                selected_time = float(sub.groupby("repeat_id")[METRIC].max().max())
            rows.append(
                {
                    "size_label": size_label,
                    "family": family,
                    "selected_algo": selected_algo,
                    "selected_time_ms": selected_time,
                    "actual_vector_size_bytes": int(meta.loc[size_label, "actual_vector_size_bytes"]),
                    "target_vector_size_bytes": int(meta.loc[size_label, "target_vector_size_bytes"]),
                }
            )
    summary = pd.DataFrame(rows)
    return add_relative_to_bidirectional_ring(summary)


def add_relative_to_bidirectional_ring(summary):
    base = (
        summary[summary["family"] == "Bidirectional Ring"]
        [["size_label", "selected_time_ms", "selected_algo"]]
        .rename(
            columns={
                "selected_time_ms": "bidirectional_ring_time_ms",
                "selected_algo": "bidirectional_ring_selected_algo",
            }
        )
    )
    if base.empty:
        raise ValueError("No Bidirectional Ring rows found. Need algo=bidirectional-ring.")
    out = summary.merge(base, on="size_label", how="left")
    out["relative_vs_bidirectional_ring_pct"] = (
        out["selected_time_ms"] / out["bidirectional_ring_time_ms"] - 1.0
    ) * 100.0
    return out


def add_bidirectional_ring_direction_annotation(ax):
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
        "BIDIRECTIONAL\nRING BETTER",
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
        "BIDIRECTIONAL\nRING WORSE",
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
        sub = summary[
            (summary["family"] == family) & summary["relative_vs_bidirectional_ring_pct"].notna()
        ].copy()
        if sub.empty:
            continue
        sub = sub.set_index("size_label")
        present = [s for s in size_order if s in sub.index]
        if not present:
            continue
        x = np.array([size_to_x[s] for s in present])
        y = sub.loc[present, "relative_vs_bidirectional_ring_pct"].to_numpy()
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
    ax.set_ylabel("Relative Completion\nTime vs. Bidirectional Ring (%)", fontsize=22, labelpad=12)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.65)
    ax.legend(loc="best", fontsize=14, frameon=True, framealpha=0.9, edgecolor="0.8")

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    if yrange == 0:
        yrange = max(abs(ymin), 1.0)
    ax.set_ylim(ymin - 0.08 * yrange, ymax + 0.08 * yrange)
    add_bidirectional_ring_direction_annotation(ax)
    fig.subplots_adjust(right=0.78, bottom=0.23)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df, meta, size_order, x_labels = load_data()
    print(f"Reading CSV: {CSV_PATH}")
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
    print("\nSelected Bidirectional Ring implementation by size:")
    print(
        mean_summary[mean_summary["family"] == "Bidirectional Ring"]
        [["size_label", "selected_algo", "selected_time_ms"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
