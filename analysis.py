#!/usr/bin/env python

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from IPython.display import display

EXPERIMENT = "base"

MODELS = Path("./input") / "models.jsonl"

OUTPUT = Path("./output") / f"{EXPERIMENT}.jsonl"

ANALYSIS = Path("./analysis")

N_BOOTSTRAP = 8_000
N_BOOTSTRAP_PER_MODEL = 2_000


def plot_arrows(ax, data, quant):
    text_elements = []
    for i, ((short, size), g) in enumerate(data.groupby(["short", "size"])):
        y = g[quant].tolist()
        label = f"{short}-{size}B"

        color = "darkgreen"
        if y[1] < y[0]:
            color = "darkred"

        ax.plot(
            [i, i],
            y,
            color=color,
            linewidth=4,
        )

        text_elements.append(label)

    ax.set_xticks(range(len(text_elements)))
    ax.set_xticklabels(text_elements, rotation=60, ha="center")
    ax.set_xlim(-0.5, len(text_elements) - 0.5)

    ax.set_ylim(0, 1)
    ax.grid(True)


def calculate_ratios(group):
    true_row = group[group["suggest_empty"] == True].iloc[0]
    false_row = group[group["suggest_empty"] == False].iloc[0]

    ratios = {}
    for col in ["INCORRECT", "DOUBT", "CORRECT", "accuracy_given_answer"]:
        ratios[f"{col}_ratio"] = (
            (true_row[col] / false_row[col] - 1) * 100 if false_row[col] != 0 else float("inf")
        )

    return pd.Series(ratios)


def bootstrap_transition_ci(pivoted: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """Bootstrap confidence intervals for INC→D and C→D transition rates.

    Resamples at the question level (preserving within-question correlation across models),
    then computes row-normalized transition rates for each bootstrap sample.

    Returns dict with 95% CI bounds for INC→D, C→D, and their difference.
    """
    questions = pivoted["question"].unique()
    n_questions = len(questions)

    question_groups = [pivoted.index[pivoted["question"] == q].to_numpy() for q in questions]

    rng = np.random.default_rng(42)
    inc_to_d = np.empty(n_bootstrap)
    c_to_d = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        selected = np.concatenate(
            [question_groups[j] for j in rng.choice(n_questions, size=n_questions, replace=True)]
        )
        sample = pivoted.loc[selected]

        grouped = sample.groupby(["base_eval", "suggest_eval"]).size()
        ct = grouped.unstack(fill_value=0)
        ct = ct.div(ct.sum(axis=1), axis=0)

        inc_to_d[i] = ct.loc["INCORRECT", "DOUBT"]
        c_to_d[i] = ct.loc["CORRECT", "DOUBT"]

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "inc_to_d": {"estimate": float(np.mean(inc_to_d)), "ci": ci(inc_to_d)},
        "c_to_d": {"estimate": float(np.mean(c_to_d)), "ci": ci(c_to_d)},
        "difference": {"estimate": float(np.mean(inc_to_d - c_to_d)), "ci": ci(inc_to_d - c_to_d)},
    }


def bootstrap_transition_ci_per_model(
    pivoted: pd.DataFrame, models_df: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP_PER_MODEL
) -> dict:
    """Bootstrap CIs per model for INC→D and C→D transition rates."""
    results = {}
    rng = np.random.default_rng(42)

    for model_name in models_df.index:
        model_data = pivoted[pivoted["model"] == model_name]
        questions = model_data["question"].unique()
        n_questions = len(questions)
        question_groups = [
            model_data.index[model_data["question"] == q].to_numpy() for q in questions
        ]

        inc_to_d = np.empty(n_bootstrap)
        c_to_d = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            selected = np.concatenate(
                [
                    question_groups[j]
                    for j in rng.choice(n_questions, size=n_questions, replace=True)
                ]
            )
            sample = model_data.loc[selected]
            grouped = sample.groupby(["base_eval", "suggest_eval"]).size()
            ct = grouped.unstack(fill_value=0)
            ct = ct.div(ct.sum(axis=1), axis=0).fillna(0)

            inc_to_d[i] = (
                ct.loc["INCORRECT", "DOUBT"]
                if "INCORRECT" in ct.index and "DOUBT" in ct.columns
                else 0.0
            )
            c_to_d[i] = (
                ct.loc["CORRECT", "DOUBT"]
                if "CORRECT" in ct.index and "DOUBT" in ct.columns
                else 0.0
            )

        def ci(arr):
            return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

        short = models_df.loc[model_name, "short"]
        size = int(models_df.loc[model_name, "size"])
        label = f"{short}-{size}B"

        results[label] = {
            "inc_to_d": {"estimate": float(np.mean(inc_to_d)), "ci": ci(inc_to_d)},
            "c_to_d": {"estimate": float(np.mean(c_to_d)), "ci": ci(c_to_d)},
            "difference": {
                "estimate": float(np.mean(inc_to_d - c_to_d)),
                "ci": ci(inc_to_d - c_to_d),
            },
        }

    return results


def _model_sort_key(label: str) -> tuple[int, int]:
    company_order = {"deepseek": 0, "gpt": 1, "llama": 2, "qwen": 3}
    short, size = label.split("-")
    return (company_order[short], int(size.replace("B", "")))


def plot_per_model_ci(per_model_ci: dict, aggregate_ci: dict, path: Path):
    """Horizontal dot plot of per-model differences (INC→D minus C→D) with aggregate."""
    labels = sorted(per_model_ci.keys(), key=_model_sort_key)
    aggregates = [per_model_ci[l]["difference"]["estimate"] for l in labels]
    cis = [per_model_ci[l]["difference"]["ci"] for l in labels]

    labels.append("All models")
    aggregates.append(aggregate_ci["difference"]["estimate"])
    cis.append(aggregate_ci["difference"]["ci"])

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, (est, (lo, hi)) in enumerate(zip(aggregates, cis)):
        y = -i
        is_aggregate = labels[i] == "All models"
        color = "#d62728" if is_aggregate else "#458588"
        marker = "D" if is_aggregate else "o"
        size = 8 if is_aggregate else 6
        ax.plot([lo, hi], [y, y], color="black", linewidth=2)
        ax.plot(est, y, marker=marker, color=color, markersize=size, zorder=5)
        ax.text(hi + 0.01, y, f"[{lo:.3f}, {hi:.3f}]", fontsize=6, va="center")

    ax.set_yticks(-np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Difference (INC→D − C→D) with 95% CI", fontsize=7)
    ax.tick_params(labelsize=7)
    sns.despine()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    with MODELS.open("r") as f:
        models = pd.DataFrame(json.loads(line) for line in f).set_index("name")

    with OUTPUT.open("r") as f:
        data = pd.DataFrame(json.loads(line) for line in f)

    grouped = data.groupby(["model", "suggest_empty"])

    metrics = []
    for (model_name, suggest_empty), group in grouped:
        total = len(group)
        evals = group["evaluation"].value_counts(normalize=True)

        answered = evals["CORRECT"] + evals["INCORRECT"]

        metrics.append(
            {
                "name": model_name,
                **models.loc[model_name, :].to_dict(),
                "suggest_empty": suggest_empty,
                **evals.to_dict(),
                "accuracy_given_answer": evals["CORRECT"] / answered if answered > 0 else None,
            }
        )

    metrics = pd.DataFrame(metrics).fillna(0.0).sort_values(by=["short", "size"])

    print(f"\n{metrics}\n")

    fig, axs = plt.subplots(2, 2, figsize=(1.8 * 4, 2 * 4), squeeze=False)
    axs = axs.flatten()

    plots = [
        ("CORRECT", "fraction of CORRECT answers"),
        ("INCORRECT", "fraction of INCORRECT answers"),
        ("DOUBT", "fraction of DOUBT answers"),
        ("accuracy_given_answer", "accuracy of answers = C/(C + INC)"),
    ]
    for i, (metric, title) in enumerate(plots):
        plot_arrows(axs[i], metrics, metric)
        axs[i].set_title(title)

    fig.savefig(ANALYSIS / "fractions.png")

    ratios = metrics.groupby(["short", "size"]).apply(calculate_ratios, include_groups=False)
    print(ratios)

    n_questions = len(data["question"].unique())
    base = metrics[~metrics["suggest_empty"]].set_index(["short", "size"])
    suggest = metrics[metrics["suggest_empty"]].set_index(["short", "size"])
    print("\n=== Absolute counts (base → suggest) ===\n")
    for short, size in ratios.index:
        row = ratios.loc[(short, size)]
        inc_b = int(round(base.loc[(short, size), "INCORRECT"] * n_questions))
        inc_s = int(round(suggest.loc[(short, size), "INCORRECT"] * n_questions))
        dbt_b = int(round(base.loc[(short, size), "DOUBT"] * n_questions))
        dbt_s = int(round(suggest.loc[(short, size), "DOUBT"] * n_questions))
        cor_b = int(round(base.loc[(short, size), "CORRECT"] * n_questions))
        cor_s = int(round(suggest.loc[(short, size), "CORRECT"] * n_questions))
        print(
            f"  {short}-{int(size)}B: INC {inc_b}→{inc_s} ({row['INCORRECT_ratio']:+.1f}%)  "
            f"DOUBT {dbt_b}→{dbt_s} ({row['DOUBT_ratio']:+.1f}%)  "
            f"COR {cor_b}→{cor_s} ({row['CORRECT_ratio']:+.1f}%)  "
            f"Δ Acc {row['accuracy_given_answer_ratio']:+.1f}%"
        )
    print()

    pivoted = (
        data.pivot_table(
            index=["model", "question"],
            columns="suggest_empty",
            values="evaluation",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={False: "base_eval", True: "suggest_eval"})
    )

    transitions_df = pivoted.groupby(["base_eval", "suggest_eval"]).size().reset_index(name="count")

    print("\n=== Transition counts (base → suggest_empty) ===\n")
    print(transitions_df)

    question_transitions = (
        pivoted.groupby(["question", "base_eval", "suggest_eval"]).size().reset_index(name="count")
    )

    question_transitions = question_transitions[
        question_transitions["base_eval"] != question_transitions["suggest_eval"]
    ].sort_values(by="count", ascending=False)

    print("\n=== Transition counts per question ===\n")
    with pd.option_context("display.max_colwidth", None):
        display(question_transitions.head(20))

    transition_counts = question_transitions.groupby(["base_eval", "suggest_eval"]).value_counts(
        ["count"]
    )
    print("\n=== Distribution of transition count per question ===\n")
    print(transition_counts)

    print("\n=== Bootstrap confidence intervals (95%) ===\n")
    ci_results = bootstrap_transition_ci(pivoted)
    for key, val in ci_results.items():
        est = val["estimate"]
        lo, hi = val["ci"]
        print(f"  {key}: {est:.3f}  [{lo:.3f}, {hi:.3f}]")
    print()

    print("\n=== Per-model bootstrap confidence intervals (95%) ===\n")
    per_model_ci = bootstrap_transition_ci_per_model(pivoted, models)
    for label in sorted(per_model_ci.keys()):
        inc = per_model_ci[label]["inc_to_d"]
        c = per_model_ci[label]["c_to_d"]
        diff = per_model_ci[label]["difference"]
        print(
            f"  {label:>12s}: INC→D {inc['estimate']:.3f} [{inc['ci'][0]:.3f}, {inc['ci'][1]:.3f}]  "
            f"C→D {c['estimate']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]  "
            f"Δ {diff['estimate']:.3f} [{diff['ci'][0]:.3f}, {diff['ci'][1]:.3f}]"
        )

    plot_per_model_ci(per_model_ci, ci_results, ANALYSIS / "per_model_ci.png")
    print("Saved per_model_ci.png\n")
