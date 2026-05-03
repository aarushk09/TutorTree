"""Analyze mass P-MCTS benchmark results and generate paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent
RUNNER_ORDER = [
    "baseline_greedy",
    "baseline_socratic_prompt",
    "pmcts_full",
    "ablation_no_reward",
    "ablation_shallow",
]
RUNNER_LABELS = {
    "baseline_greedy": "Greedy",
    "baseline_socratic_prompt": "Socratic Prompt",
    "pmcts_full": "P-MCTS Full",
    "ablation_no_reward": "No Reward",
    "ablation_shallow": "Shallow",
}


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "scenario_id",
        "domain",
        "runner_type",
        "selected_action",
        "student_state",
        "pedagogical_reward",
        "total_tokens_used",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results CSV missing required columns: {sorted(missing)}")
    return df


def summarize_by_runner(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("runner_type", as_index=False)
        .agg(
            avg_reward=("pedagogical_reward", "mean"),
            total_tokens=("total_tokens_used", "sum"),
            interactions=("pedagogical_reward", "size"),
        )
        .assign(runner_label=lambda frame: frame["runner_type"].map(RUNNER_LABELS))
    )
    summary["runner_type"] = pd.Categorical(summary["runner_type"], RUNNER_ORDER, ordered=True)
    return summary.sort_values("runner_type")


def summarize_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["domain", "runner_type"], as_index=False)
        .agg(avg_reward=("pedagogical_reward", "mean"))
        .assign(runner_label=lambda frame: frame["runner_type"].map(RUNNER_LABELS))
    )
    grouped["runner_type"] = pd.Categorical(grouped["runner_type"], RUNNER_ORDER, ordered=True)
    return grouped.sort_values(["domain", "runner_type"])


def hardest_domain(df: pd.DataFrame) -> tuple[str, float]:
    domain_scores = df.groupby("domain")["pedagogical_reward"].mean().sort_values()
    return str(domain_scores.index[0]), float(domain_scores.iloc[0])


def plot_reward_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(8, 4.8), dpi=180)
    ax = sns.barplot(
        data=summary,
        x="runner_label",
        y="avg_reward",
        hue="runner_label",
        palette="viridis",
        legend=False,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Average Pedagogical Reward")
    ax.set_title("Average Reward Across 100 Benchmark Scenarios")
    ax.set_ylim(-1.05, 1.05)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pareto_frontier(summary: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(7.2, 5), dpi=180)
    ax = sns.scatterplot(
        data=summary,
        x="total_tokens",
        y="avg_reward",
        hue="runner_label",
        s=110,
        palette="deep",
    )
    for _, row in summary.iterrows():
        ax.annotate(
            row["runner_label"],
            (row["total_tokens"], row["avg_reward"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Total Tokens Used")
    ax.set_ylabel("Average Pedagogical Reward")
    ax.set_title("Compute Cost vs. Pedagogical Reward")
    ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_domain_breakdown(domain_summary: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(9, 5), dpi=180)
    ax = sns.barplot(
        data=domain_summary,
        x="domain",
        y="avg_reward",
        hue="runner_label",
        palette="muted",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Average Pedagogical Reward")
    ax.set_title("Average Reward by Domain")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(title="Runner", loc="lower right", fontsize=8, title_fontsize=8)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_summary(summary: pd.DataFrame, domain_summary: pd.DataFrame, df: pd.DataFrame) -> None:
    print("Runner Summary")
    print("=" * 62)
    print(f"{'Runner':24} {'Avg Reward':>12} {'Total Tokens':>14} {'N':>6}")
    print("-" * 62)
    for _, row in summary.iterrows():
        print(
            f"{row['runner_label']:24} "
            f"{row['avg_reward']:12.3f} "
            f"{int(row['total_tokens']):14d} "
            f"{int(row['interactions']):6d}"
        )

    print("\nDomain Breakdown")
    print("=" * 62)
    pivot = domain_summary.pivot_table(
        index="domain",
        columns="runner_label",
        values="avg_reward",
        aggfunc="mean",
    )
    print(pivot.round(3).to_string())
    domain, value = hardest_domain(df)
    print(f"\nHardest domain by overall average reward: {domain} ({value:.3f})")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "mass_experiment_results.csv")
    parser.add_argument("--reward-plot", type=Path, default=PROJECT_ROOT / "reward_comparison.png")
    parser.add_argument("--pareto-plot", type=Path, default=PROJECT_ROOT / "pareto_frontier.png")
    parser.add_argument("--domain-plot", type=Path, default=PROJECT_ROOT / "domain_breakdown.png")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    df = load_results(args.input)
    summary = summarize_by_runner(df)
    domain_summary = summarize_by_domain(df)

    plot_reward_comparison(summary, args.reward_plot)
    plot_pareto_frontier(summary, args.pareto_plot)
    plot_domain_breakdown(domain_summary, args.domain_plot)

    print_summary(summary, domain_summary, df)
    print("\nSaved artifacts:")
    print(f"- {args.reward_plot.name}")
    print(f"- {args.pareto_plot.name}")
    print(f"- {args.domain_plot.name}")


if __name__ == "__main__":
    main()
