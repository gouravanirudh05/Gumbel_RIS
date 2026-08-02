#!/usr/bin/env python3
"""Parse saved experiment logs and generate publication-ready CSVs/plots.

The expected output tree is:

    outputs/<Approach>/<N_RIS>/<spacing>.txt

where spacing is one of lambda_4, lambda_2, or lambda. Empty or incomplete logs
are skipped automatically.
"""

from __future__ import annotations

import argparse
import re
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


APPROACHES = ("Greedy", "Gumbel", "Transformer")
N_RIS_VALUES = (16, 64, 256)
SPACING_ORDER = ("lambda_4", "lambda_2", "lambda")
SPACING_TO_LAMBDA = {"lambda_4": 0.25, "lambda_2": 0.50, "lambda": 1.00}
SPACING_LABEL = {"lambda_4": r"$\lambda/4$", "lambda_2": r"$\lambda/2$", "lambda": r"$\lambda$"}

ROW_RE = re.compile(
    r"^(?P<training>Standard \(no MC\)|MC-aware)\s*\|\s*"
    r"(?P<eval_standard>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"(?P<eval_mc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def spacing_sort_key(spacing: str) -> float:
    return SPACING_TO_LAMBDA.get(spacing, float("inf"))


def parse_result_file(path: Path, approach: str, n_ris: int, spacing: str) -> list[dict]:
    """Return tidy rows for one log file.

    Each complete result line contributes two rows: evaluation without mutual
    coupling and evaluation with mutual coupling.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []

    rows: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        match = ROW_RE.match(line.strip())
        if not match:
            continue

        training_label = match.group("training")
        training_policy = "standard_no_mc" if training_label.startswith("Standard") else "mc_aware"
        values = {
            "no_coupling_eval": float(match.group("eval_standard")),
            "with_coupling_eval": float(match.group("eval_mc")),
        }

        for eval_condition, channel_gain in values.items():
            rows.append(
                {
                    "approach": approach,
                    "n_ris": n_ris,
                    "spacing_key": spacing,
                    "spacing_lambda": SPACING_TO_LAMBDA[spacing],
                    "spacing_label": SPACING_LABEL[spacing].replace("$", ""),
                    "training_policy": training_policy,
                    "evaluation_condition": eval_condition,
                    "channel_gain": channel_gain,
                    "source_file": str(path),
                }
            )

    return rows


def build_results(outputs_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for approach in APPROACHES:
        for n_ris in N_RIS_VALUES:
            for spacing in SPACING_ORDER:
                path = outputs_dir / approach / str(n_ris) / f"{spacing}.txt"
                rows.extend(parse_result_file(path, approach, n_ris, spacing))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    policy_order = {"standard_no_mc": 0, "mc_aware": 1}
    eval_order = {"no_coupling_eval": 0, "with_coupling_eval": 1}
    df = df.sort_values(
        by=["approach", "n_ris", "spacing_lambda", "training_policy", "evaluation_condition"],
        key=lambda col: col.map(policy_order).fillna(col.map(eval_order)).fillna(col)
        if col.name in ("training_policy", "evaluation_condition")
        else col,
    ).reset_index(drop=True)
    return df


def make_wide_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    wide = (
        df.pivot_table(
            index=["approach", "n_ris", "spacing_key", "spacing_lambda", "spacing_label"],
            columns=["training_policy", "evaluation_condition"],
            values="channel_gain",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns = [
        "_".join(str(part) for part in col if part).strip("_") if isinstance(col, tuple) else str(col)
        for col in wide.columns
    ]

    std_mc = "standard_no_mc_with_coupling_eval"
    aware_mc = "mc_aware_with_coupling_eval"
    if std_mc in wide.columns and aware_mc in wide.columns:
        wide["mc_aware_gain_delta"] = wide[aware_mc] - wide[std_mc]
        wide["mc_aware_gain_delta_percent"] = 100.0 * wide["mc_aware_gain_delta"] / wide[std_mc]

    return wide.sort_values(["approach", "n_ris", "spacing_lambda"]).reset_index(drop=True)


def make_improvement_summary(wide: pd.DataFrame) -> pd.DataFrame:
    """Return the main paper metric: MC-aware improvement under MC evaluation."""
    required = {
        "approach",
        "n_ris",
        "spacing_key",
        "spacing_lambda",
        "spacing_label",
        "standard_no_mc_with_coupling_eval",
        "mc_aware_with_coupling_eval",
        "mc_aware_gain_delta",
        "mc_aware_gain_delta_percent",
    }
    missing = required.difference(wide.columns)
    if missing:
        raise ValueError(f"Missing required summary columns: {sorted(missing)}")

    improvement = wide[
        [
            "approach",
            "n_ris",
            "spacing_key",
            "spacing_lambda",
            "spacing_label",
            "standard_no_mc_with_coupling_eval",
            "mc_aware_with_coupling_eval",
            "mc_aware_gain_delta",
            "mc_aware_gain_delta_percent",
        ]
    ].copy()
    improvement = improvement.rename(
        columns={
            "standard_no_mc_with_coupling_eval": "baseline_gain_with_mc",
            "mc_aware_with_coupling_eval": "mc_aware_gain_with_mc",
            "mc_aware_gain_delta": "absolute_improvement",
            "mc_aware_gain_delta_percent": "relative_improvement_percent",
        }
    )
    return improvement.sort_values(["approach", "n_ris", "spacing_lambda"]).reset_index(drop=True)


def apply_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.linewidth": 0.9,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_lines(
    ax: plt.Axes,
    subset: pd.DataFrame,
    x_col: str,
    x_values: list[float] | list[int],
    show_no_coupling_reference: bool,
) -> None:
    colors = {"Greedy": "#1b4f72", "Gumbel": "#b03a2e", "Transformer": "#196f3d"}
    markers = {"Greedy": "o", "Gumbel": "s", "Transformer": "^"}
    policies = [
        ("standard_no_mc", "with_coupling_eval", "--", "Std train, eval w/ MC"),
        ("mc_aware", "with_coupling_eval", "-", "MC-aware, eval w/ MC"),
    ]

    if show_no_coupling_reference:
        policies.insert(0, ("standard_no_mc", "no_coupling_eval", ":", "No-coupling ref."))

    for approach in APPROACHES:
        approach_df = subset[subset["approach"] == approach]
        if approach_df.empty:
            continue

        for policy, eval_condition, linestyle, label_suffix in policies:
            line_df = approach_df[
                (approach_df["training_policy"] == policy)
                & (approach_df["evaluation_condition"] == eval_condition)
            ].sort_values(x_col)
            if line_df.empty:
                continue

            ax.plot(
                line_df[x_col],
                line_df["channel_gain"],
                color=colors[approach],
                marker=markers[approach],
                linestyle=linestyle,
                label=f"{approach}: {label_suffix}",
            )

    ax.set_xticks(x_values)
    ax.tick_params(direction="in", length=4, width=0.8)
    ax.grid(True, which="major", linestyle="--", linewidth=0.55, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.40, alpha=0.35)


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def generate_spacing_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    for n_ris in sorted(df["n_ris"].unique()):
        subset = df[df["n_ris"] == n_ris]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        plot_lines(
            ax=ax,
            subset=subset,
            x_col="spacing_lambda",
            x_values=[0.25, 0.50, 1.00],
            show_no_coupling_reference=True,
        )
        ax.set_title(rf"Channel Gain vs RIS Element Spacing ($N_{{RIS}}={n_ris}$)")
        ax.set_xlabel(r"Inter-element spacing")
        ax.set_ylabel("Average channel gain")
        ax.set_xticklabels([SPACING_LABEL[s] for s in SPACING_ORDER])
        ax.legend(loc="best", frameon=True, framealpha=0.92, ncol=1)
        fig.tight_layout()
        save_figure(fig, plots_dir / f"channel_gain_vs_spacing_N{n_ris}")


def generate_nris_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    for spacing in SPACING_ORDER:
        subset = df[df["spacing_key"] == spacing]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        plot_lines(
            ax=ax,
            subset=subset,
            x_col="n_ris",
            x_values=[16, 64, 256],
            show_no_coupling_reference=True,
        )
        ax.set_title(rf"Channel Gain vs Number of RIS Elements ({SPACING_LABEL[spacing]} spacing)")
        ax.set_xlabel(r"Number of RIS elements, $N_{RIS}$")
        ax.set_ylabel("Average channel gain")
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256])
        ax.set_xticklabels(["16", "64", "256"])
        ax.legend(loc="best", frameon=True, framealpha=0.92, ncol=1)
        fig.tight_layout()
        save_figure(fig, plots_dir / f"channel_gain_vs_nris_{spacing}")


def generate_compact_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    """Create cleaner figures using only the real-world MC-aware metric."""
    compact = df[
        (df["training_policy"] == "mc_aware") & (df["evaluation_condition"] == "with_coupling_eval")
    ].copy()
    if compact.empty:
        return

    colors = {"Greedy": "#1b4f72", "Gumbel": "#b03a2e", "Transformer": "#196f3d"}
    markers = {"Greedy": "o", "Gumbel": "s", "Transformer": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), sharey=False)
    for ax, n_ris in zip(axes, sorted(compact["n_ris"].unique())):
        subset = compact[compact["n_ris"] == n_ris]
        for approach in APPROACHES:
            line_df = subset[subset["approach"] == approach].sort_values("spacing_lambda")
            if line_df.empty:
                continue
            ax.plot(
                line_df["spacing_lambda"],
                line_df["channel_gain"],
                color=colors[approach],
                marker=markers[approach],
                label=approach,
            )
        ax.set_title(rf"$N_{{RIS}}={n_ris}$")
        ax.set_xlabel(r"Spacing")
        ax.set_xticks([0.25, 0.50, 1.00])
        ax.set_xticklabels([SPACING_LABEL[s] for s in SPACING_ORDER])
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.55)
    axes[0].set_ylabel("Average channel gain")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, plots_dir / "compact_mcaware_gain_vs_spacing")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), sharey=False)
    for ax, spacing in zip(axes, SPACING_ORDER):
        subset = compact[compact["spacing_key"] == spacing]
        for approach in APPROACHES:
            line_df = subset[subset["approach"] == approach].sort_values("n_ris")
            if line_df.empty:
                continue
            ax.plot(
                line_df["n_ris"],
                line_df["channel_gain"],
                color=colors[approach],
                marker=markers[approach],
                label=approach,
            )
        ax.set_title(rf"{SPACING_LABEL[spacing]} spacing")
        ax.set_xlabel(r"$N_{RIS}$")
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256])
        ax.set_xticklabels(["16", "64", "256"])
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.55)
    axes[0].set_ylabel("Average channel gain")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, plots_dir / "compact_mcaware_gain_vs_nris")


def style_improvement_axis(ax: plt.Axes) -> None:
    ax.axhline(0.0, color="#202020", linewidth=0.9, linestyle="-", alpha=0.75)
    ax.tick_params(direction="in", length=4, width=0.8)
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.55, alpha=0.55)
    ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.40, alpha=0.35)


def generate_improvement_plots(improvement: pd.DataFrame, plots_dir: Path) -> None:
    """Plot relative gain of MC-aware optimization over the standard baseline."""
    if improvement.empty:
        return

    colors = {"Greedy": "#1b4f72", "Gumbel": "#b03a2e", "Transformer": "#196f3d"}
    markers = {"Greedy": "o", "Gumbel": "s", "Transformer": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
    for ax, n_ris in zip(axes, sorted(improvement["n_ris"].unique())):
        subset = improvement[improvement["n_ris"] == n_ris]
        for approach in APPROACHES:
            line_df = subset[subset["approach"] == approach].sort_values("spacing_lambda")
            if line_df.empty:
                continue
            ax.plot(
                line_df["spacing_lambda"],
                line_df["relative_improvement_percent"],
                color=colors[approach],
                marker=markers[approach],
                label=approach,
            )
        ax.set_title(rf"$N_{{RIS}}={n_ris}$")
        ax.set_xlabel(r"Spacing")
        ax.set_xticks([0.25, 0.50, 1.00])
        ax.set_xticklabels([SPACING_LABEL[s] for s in SPACING_ORDER])
        style_improvement_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.supylabel("MC-aware improvement over standard (%)", x=0.01)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True)
    fig.tight_layout(rect=[0.04, 0.08, 1, 1])
    save_figure(fig, plots_dir / "mcaware_relative_improvement_vs_spacing")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
    for ax, spacing in zip(axes, SPACING_ORDER):
        subset = improvement[improvement["spacing_key"] == spacing]
        for approach in APPROACHES:
            line_df = subset[subset["approach"] == approach].sort_values("n_ris")
            if line_df.empty:
                continue
            ax.plot(
                line_df["n_ris"],
                line_df["relative_improvement_percent"],
                color=colors[approach],
                marker=markers[approach],
                label=approach,
            )
        ax.set_title(rf"{SPACING_LABEL[spacing]} spacing")
        ax.set_xlabel(r"$N_{RIS}$")
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256])
        ax.set_xticklabels(["16", "64", "256"])
        style_improvement_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.supylabel("MC-aware improvement over standard (%)", x=0.01)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True)
    fig.tight_layout(rect=[0.04, 0.08, 1, 1])
    save_figure(fig, plots_dir / "mcaware_relative_improvement_vs_nris")

    for n_ris in sorted(improvement["n_ris"].unique()):
        subset = improvement[improvement["n_ris"] == n_ris]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for approach in APPROACHES:
            line_df = subset[subset["approach"] == approach].sort_values("spacing_lambda")
            if line_df.empty:
                continue
            ax.plot(
                line_df["spacing_lambda"],
                line_df["relative_improvement_percent"],
                color=colors[approach],
                marker=markers[approach],
                label=approach,
            )
        ax.set_title(rf"MC-aware Improvement vs Spacing ($N_{{RIS}}={n_ris}$)")
        ax.set_xlabel(r"Inter-element spacing")
        ax.set_ylabel("Relative improvement (%)")
        ax.set_xticks([0.25, 0.50, 1.00])
        ax.set_xticklabels([SPACING_LABEL[s] for s in SPACING_ORDER])
        style_improvement_axis(ax)
        ax.legend(loc="best", frameon=True, framealpha=0.92)
        fig.tight_layout()
        save_figure(fig, plots_dir / f"mcaware_relative_improvement_vs_spacing_N{n_ris}")

    for spacing in SPACING_ORDER:
        subset = improvement[improvement["spacing_key"] == spacing]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for approach in APPROACHES:
            line_df = subset[subset["approach"] == approach].sort_values("n_ris")
            if line_df.empty:
                continue
            ax.plot(
                line_df["n_ris"],
                line_df["relative_improvement_percent"],
                color=colors[approach],
                marker=markers[approach],
                label=approach,
            )
        ax.set_title(rf"MC-aware Improvement vs RIS Elements ({SPACING_LABEL[spacing]} spacing)")
        ax.set_xlabel(r"Number of RIS elements, $N_{RIS}$")
        ax.set_ylabel("Relative improvement (%)")
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256])
        ax.set_xticklabels(["16", "64", "256"])
        style_improvement_axis(ax)
        ax.legend(loc="best", frameon=True, framealpha=0.92)
        fig.tight_layout()
        save_figure(fig, plots_dir / f"mcaware_relative_improvement_vs_nris_{spacing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = build_results(args.outputs_dir)
    if df.empty:
        raise SystemExit(f"No complete result rows found under {args.outputs_dir}")

    tidy_csv = args.results_dir / "channel_gain_results_tidy.csv"
    wide_csv = args.results_dir / "channel_gain_results_summary.csv"
    improvement_csv = args.results_dir / "mcaware_relative_improvement.csv"
    df.to_csv(tidy_csv, index=False)
    wide = make_wide_summary(df)
    improvement = make_improvement_summary(wide)
    wide.to_csv(wide_csv, index=False)
    improvement.to_csv(improvement_csv, index=False)

    apply_plot_style()
    generate_spacing_plots(df, plots_dir)
    generate_nris_plots(df, plots_dir)
    generate_compact_plots(df, plots_dir)
    generate_improvement_plots(improvement, plots_dir)

    print(f"Parsed {len(df)} tidy rows from {df['source_file'].nunique()} log files.")
    print(f"Wrote {tidy_csv}")
    print(f"Wrote {wide_csv}")
    print(f"Wrote {improvement_csv}")
    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
