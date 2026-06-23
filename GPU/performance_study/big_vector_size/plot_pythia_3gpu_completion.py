#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CSV = Path("pythia_3gpu_completion_results.csv")
METRIC_TOTAL = "total_training_time_sec"
METRIC_STEP = "avg_step_time_ms"
BASELINE_ALGO = "trivance-bandwidth"

ALGO_LABELS = {
    "builtin": "BuiltIn",
    "ring": "Ring",
    "trivance-bandwidth": "Trivance-B",
}
ALGO_ORDER = ["builtin", "ring", "trivance-bandwidth"]

STYLES = {
    "builtin": {"marker": "D", "color": "#ff7f0e"},
    "ring": {"marker": "o", "color": "#1f77b4"},
    "trivance-bandwidth": {"marker": "s", "color": "#9467bd"},
}

BYTES_PER_DTYPE = {
    "fp32": 4,
    "float32": 4,
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
}


def format_gib(x: float) -> str:
    return f"{x:.2f} GiB"


def short_model_name(model_name: str) -> str:
    name = model_name.split("/")[-1]
    return name.replace("pythia-", "Pythia-")


def load_and_summarize(csv_path: Path):
    df = pd.read_csv(csv_path)

    if "accepted" in df.columns:
        df = df[df["accepted"].astype(str).str.lower().isin(["true", "1"])]

    required = [
        "model_name",
        "model_label",
        "algo",
        "dtype",
        "steps",
        "batch_size",
        "block_size",
        "bucket_cap_mb",
        "trainable_params",
        METRIC_TOTAL,
        METRIC_STEP,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; columns={list(df.columns)}")

    df = df[df["algo"].isin(ALGO_ORDER)].copy()
    if df.empty:
        raise ValueError("No supported algorithms found. Expected builtin, ring, trivance-bandwidth.")

    df["param_bytes"] = df["dtype"].map(BYTES_PER_DTYPE).fillna(4).astype(int)
    df["message_size_bytes"] = df["trainable_params"].astype(np.int64) * df["param_bytes"].astype(np.int64)
    df["message_size_mib"] = df["message_size_bytes"] / (1024 ** 2)
    df["message_size_gib"] = df["message_size_bytes"] / (1024 ** 3)

    group_cols = [
        "model_name",
        "model_label",
        "algo",
        "dtype",
        "steps",
        "batch_size",
        "block_size",
        "bucket_cap_mb",
        "trainable_params",
        "message_size_bytes",
        "message_size_mib",
        "message_size_gib",
    ]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            runs=(METRIC_TOTAL, "count"),
            mean_total_sec=(METRIC_TOTAL, "mean"),
            std_total_sec=(METRIC_TOTAL, "std"),
            mean_step_ms=(METRIC_STEP, "mean"),
            std_step_ms=(METRIC_STEP, "std"),
            checksum_max=("parameter_checksum_max_diff", "max") if "parameter_checksum_max_diff" in df.columns else (METRIC_TOTAL, "size"),
        )
        .reset_index()
    )
    summary["std_total_sec"] = summary["std_total_sec"].fillna(0.0)
    summary["std_step_ms"] = summary["std_step_ms"].fillna(0.0)
    summary["algo_label"] = summary["algo"].map(ALGO_LABELS).fillna(summary["algo"])
    summary["model_short"] = summary["model_name"].map(short_model_name)
    summary["algo_order"] = summary["algo"].map({a: i for i, a in enumerate(ALGO_ORDER)})
    summary = summary.sort_values(["trainable_params", "algo_order"]).drop(columns=["algo_order"])

    baseline = summary[summary["algo"] == BASELINE_ALGO][["model_name", "mean_total_sec", "mean_step_ms"]]
    if baseline.empty:
        raise ValueError(f"No baseline rows found for {BASELINE_ALGO}.")
    baseline = baseline.rename(columns={"mean_total_sec": "trivance_total_sec", "mean_step_ms": "trivance_step_ms"})

    summary = summary.merge(baseline, on="model_name", how="left")
    summary["relative_total_vs_trivance_pct"] = (summary["mean_total_sec"] / summary["trivance_total_sec"] - 1.0) * 100.0
    summary["relative_step_vs_trivance_pct"] = (summary["mean_step_ms"] / summary["trivance_step_ms"] - 1.0) * 100.0

    model_meta = (
        summary.drop_duplicates("model_name")
        .sort_values("trainable_params")
        [["model_name", "model_short", "trainable_params", "message_size_bytes", "message_size_mib", "message_size_gib"]]
    )
    model_order = model_meta["model_name"].tolist()
    x_labels = [
        f"{row.model_short}\n{format_gib(row.message_size_gib)}"
        for row in model_meta.itertuples(index=False)
    ]

    return df, summary, model_meta, model_order, x_labels


def plot_absolute(summary, model_order, x_labels, y_col, y_err_col, ylabel, out_path, title=""):
    x = np.arange(len(model_order))
    fig, ax = plt.subplots(figsize=(10.8, 6.3))

    for algo in ALGO_ORDER:
        sub = summary[summary["algo"] == algo].set_index("model_name")
        present = [m for m in model_order if m in sub.index]
        if not present:
            continue
        xs = np.array([model_order.index(m) for m in present])
        y = sub.loc[present, y_col].to_numpy(dtype=float)
        yerr = sub.loc[present, y_err_col].to_numpy(dtype=float)
        st = STYLES.get(algo, {"marker": "o", "color": None})
        ax.errorbar(
            xs,
            y,
            yerr=yerr,
            marker=st["marker"],
            color=st["color"],
            linewidth=2.6,
            markersize=8,
            capsize=4,
            label=ALGO_LABELS.get(algo, algo),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=12)
    ax.set_xlabel("Model / gradient message size", fontsize=18, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=12)
    if title:
        ax.set_title(title, fontsize=16, pad=12)
    ax.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.65)
    ax.legend(loc="best", fontsize=13, frameon=True, framealpha=0.92, edgecolor="0.8")
    ax.tick_params(axis="y", labelsize=12)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_relative(summary, model_order, x_labels, out_path, title=""):
    x = np.arange(len(model_order))
    fig, ax = plt.subplots(figsize=(10.8, 6.3))

    # Plot non-baseline families; baseline is the zero line.
    for algo in ["builtin", "ring"]:
        sub = summary[summary["algo"] == algo].set_index("model_name")
        present = [m for m in model_order if m in sub.index]
        if not present:
            continue
        xs = np.array([model_order.index(m) for m in present])
        y = sub.loc[present, "relative_total_vs_trivance_pct"].to_numpy(dtype=float)
        st = STYLES.get(algo, {"marker": "o", "color": None})
        ax.plot(
            xs,
            y,
            marker=st["marker"],
            color=st["color"],
            linewidth=2.6,
            markersize=8,
            label=ALGO_LABELS.get(algo, algo),
        )

    ax.axhline(0, linestyle="--", linewidth=1.4, color="0.35", alpha=0.85, label="Trivance-B")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=12)
    ax.set_xlabel("Model size", fontsize=18, labelpad=12)
    ax.set_ylabel("Relative completion time\nvs. Trivance-B (%)", fontsize=18, labelpad=12)
    if title:
        ax.set_title(title, fontsize=16, pad=12)
    ax.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.65)
    ax.legend(loc="best", fontsize=13, frameon=True, framealpha=0.92, edgecolor="0.8")
    ax.tick_params(axis="y", labelsize=12)
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax != ymin else 1.0
    ax.set_ylim(ymin - 0.10 * yrange, ymax + 0.10 * yrange)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default=str(DEFAULT_CSV))
    parser.add_argument("--out-prefix", default="pythia_3gpu")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_prefix = Path(args.out_prefix)

    df, summary, model_meta, model_order, x_labels = load_and_summarize(csv_path)

    summary_csv = Path(f"{out_prefix}_completion_summary.csv")
    message_csv = Path(f"{out_prefix}_message_sizes.csv")
    summary.to_csv(summary_csv, index=False)
    model_meta.to_csv(message_csv, index=False)

    abs_total_pdf = Path(f"{out_prefix}_absolute_total_completion_time.pdf")
    abs_total_png = Path(f"{out_prefix}_absolute_total_completion_time.png")
    abs_step_pdf = Path(f"{out_prefix}_absolute_avg_step_time.pdf")
    abs_step_png = Path(f"{out_prefix}_absolute_avg_step_time.png")
    rel_pdf = Path(f"{out_prefix}_relative_vs_trivance.pdf")
    rel_png = Path(f"{out_prefix}_relative_vs_trivance.png")

    plot_absolute(
        summary,
        model_order,
        x_labels,
        y_col="mean_total_sec",
        y_err_col="std_total_sec",
        ylabel="Mean total completion time (s)",
        out_path=abs_total_pdf,
    )
    plot_absolute(
        summary,
        model_order,
        x_labels,
        y_col="mean_total_sec",
        y_err_col="std_total_sec",
        ylabel="Mean total completion time (s)",
        out_path=abs_total_png,
    )

    plot_absolute(
        summary,
        model_order,
        x_labels,
        y_col="mean_step_ms",
        y_err_col="std_step_ms",
        ylabel="Mean step time (ms)",
        out_path=abs_step_pdf,
    )
    plot_absolute(
        summary,
        model_order,
        x_labels,
        y_col="mean_step_ms",
        y_err_col="std_step_ms",
        ylabel="Mean step time (ms)",
        out_path=abs_step_png,
    )

    plot_relative(summary, model_order, x_labels, rel_pdf)
    plot_relative(summary, model_order, x_labels, rel_png)

    print("Message sizes:")
    print(model_meta.to_string(index=False))
    print("\nSummary:")
    print(summary[["model_name", "algo", "runs", "mean_total_sec", "std_total_sec", "mean_step_ms", "std_step_ms", "message_size_gib", "relative_total_vs_trivance_pct", "checksum_max"]].to_string(index=False))
    print("\nSaved files:")
    for p in [summary_csv, message_csv, abs_total_pdf, abs_total_png, abs_step_pdf, abs_step_png, rel_pdf, rel_png]:
        print(p)


if __name__ == "__main__":
    main()
