from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["never", "random", "always", "fliponline"]
METHOD_COLORS = {
    "never": "#6B7280",
    "random": "#F59E0B",
    "always": "#10B981",
    "fliponline": "#2563EB",
}


def create_plots(summary: pd.DataFrame, rounds: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    networks = list(summary["network"].drop_duplicates())
    figure, axes = plt.subplots(
        1, len(networks), figsize=(6 * len(networks), 4), sharey=True
    )
    if len(networks) == 1:
        axes = [axes]
    for axis, network in zip(axes, networks):
        subset = summary[summary["network"] == network].copy()
        subset["method"] = pd.Categorical(subset["method"], METHOD_ORDER, ordered=True)
        subset = subset.sort_values("method")
        axis.bar(
            subset["method"].astype(str),
            subset["final_auc"],
            color=[METHOD_COLORS[str(method)] for method in subset["method"]],
        )
        axis.set_title(f"{network.capitalize()} network")
        axis.set_xlabel("Method")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Patient-weighted ROC AUC")
    figure.tight_layout()
    figure.savefig(output_dir / "final_auc.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(networks), figsize=(6 * len(networks), 4), sharey=True
    )
    if len(networks) == 1:
        axes = [axes]
    for axis, network in zip(axes, networks):
        for method in METHOD_ORDER:
            subset = rounds[
                (rounds["network"] == network) & (rounds["method"] == method)
            ]
            if subset.empty:
                continue
            axis.plot(
                subset["round"],
                subset["auc"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=METHOD_COLORS[method],
                label=method,
            )
        axis.set_title(f"{network.capitalize()} network")
        axis.set_xlabel("FL round")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Patient-weighted ROC AUC")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "auc_over_rounds.png", dpi=180)
    plt.close(figure)

    relative_metrics = [
        ("delta_auc_vs_never", "Final AUC difference"),
        ("y1_gain_vs_never_pct", "F1/time gain (%)"),
        ("time_reduction_vs_never_pct", "Round-time reduction (%)"),
    ]
    figure, axes = plt.subplots(
        len(networks),
        len(relative_metrics),
        figsize=(5 * len(relative_metrics), 3.5 * len(networks)),
        squeeze=False,
    )
    for row_index, network in enumerate(networks):
        subset = summary[
            (summary["network"] == network) & (summary["method"] != "never")
        ].copy()
        subset["method"] = pd.Categorical(subset["method"], METHOD_ORDER, ordered=True)
        subset = subset.sort_values("method")
        for column_index, (metric, title) in enumerate(relative_metrics):
            axis = axes[row_index, column_index]
            values = subset[metric]
            if metric == "delta_auc_vs_never":
                values = values * 100
                title = "Final AUC difference (points)"
            axis.bar(
                subset["method"].astype(str),
                values,
                color=[METHOD_COLORS[str(method)] for method in subset["method"]],
            )
            axis.axhline(0, color="#111827", linewidth=0.8)
            axis.set_title(f"{network.capitalize()}: {title}")
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "relative_tradeoffs.png", dpi=180)
    plt.close(figure)
