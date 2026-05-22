"""
Student Performance Analyzer — full analysis pipeline.

Loads the synthetic dataset produced by generate_data.py, cleans it,
computes summary statistics and correlations, writes six charts to
output/, and prints a console summary.

Run:
    python generate_data.py   # once, to create students_raw.csv
    python main.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

DATA_FILE = "students_raw.csv"
OUTPUT_DIR = Path("output")
PASS_THRESHOLD = 2.0

NUMERIC_COLS = [
    "attendance_rate",
    "study_hours_per_week",
    "math_score",
    "science_score",
    "english_score",
    "history_score",
    "gpa",
]


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------

def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python generate_data.py` first."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Strip noise from the raw dataset: dupes, bad types, missing values."""
    before = len(df)
    df = df.drop_duplicates(subset="student_id", keep="first").copy()
    print(f"  Removed {before - len(df)} duplicate rows")

    # attendance_rate has some "92.5%" strings — strip the % and coerce to float.
    df["attendance_rate"] = (
        df["attendance_rate"]
        .astype(str)
        .str.rstrip("%")
        .replace({"nan": np.nan, "": np.nan})
        .astype(float)
    )

    # Fill numeric nulls with the column median (robust to outliers).
    for col in ["attendance_rate", "study_hours_per_week", "english_score"]:
        if df[col].isna().any():
            median = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median)
            print(f"  Filled {n_missing} nulls in '{col}' with median ({median:.1f})")

    # Categorical nulls: fall back to the mode.
    for col in ["parent_education_level", "internet_access", "name"]:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True).iloc[0]
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(mode)
            print(f"  Filled {n_missing} nulls in '{col}' with mode ('{mode}')")

    df["study_hours_per_week"] = df["study_hours_per_week"].astype(int)
    df["grade_level"] = df["grade_level"].astype(int)

    # Recompute the pass flag so it's always consistent with the cleaned GPA.
    df["passed"] = np.where(df["gpa"] >= PASS_THRESHOLD, "Yes", "No")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def group_stats(df: pd.DataFrame, by: str) -> pd.DataFrame:
    return df.groupby(by)["gpa"].agg(["mean", "median", "std", "count"]).round(3)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC_COLS].corr().round(3)


def top_gpa_correlations(corr: pd.DataFrame, k: int = 3) -> list[tuple[str, float]]:
    gpa_corr = corr["gpa"].drop("gpa").sort_values(key=abs, ascending=False)
    return list(gpa_corr.head(k).items())


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_gpa_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["gpa"], bins=20, color="#4C72B0", edgecolor="white")
    ax.axvline(PASS_THRESHOLD, color="crimson", linestyle="--", label=f"Pass ≥ {PASS_THRESHOLD}")
    ax.set_title("GPA Distribution")
    ax.set_xlabel("GPA")
    ax.set_ylabel("Number of students")
    ax.legend()
    _save(fig, "gpa_distribution.png")


def plot_avg_gpa_by_grade(df: pd.DataFrame) -> None:
    means = df.groupby("grade_level")["gpa"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(means.index.astype(str), means.values, color="#55A868")
    ax.set_title("Average GPA by Grade Level")
    ax.set_xlabel("Grade level")
    ax.set_ylabel("Average GPA")
    ax.set_ylim(0, 4)
    for x, y in zip(means.index.astype(str), means.values):
        ax.text(x, y + 0.05, f"{y:.2f}", ha="center")
    _save(fig, "avg_gpa_by_grade.png")


def plot_study_vs_gpa(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["study_hours_per_week"], df["gpa"], alpha=0.45, color="#C44E52")
    # Linear trend line.
    m, b = np.polyfit(df["study_hours_per_week"], df["gpa"], 1)
    xs = np.linspace(df["study_hours_per_week"].min(), df["study_hours_per_week"].max(), 50)
    ax.plot(xs, m * xs + b, color="black", linewidth=1.5, label=f"trend: y={m:.3f}x+{b:.2f}")
    ax.set_title("Study Hours vs GPA")
    ax.set_xlabel("Study hours per week")
    ax.set_ylabel("GPA")
    ax.legend()
    _save(fig, "study_vs_gpa.png")


def plot_attendance_vs_gpa(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["attendance_rate"], df["gpa"], alpha=0.45, color="#8172B2")
    m, b = np.polyfit(df["attendance_rate"], df["gpa"], 1)
    xs = np.linspace(df["attendance_rate"].min(), df["attendance_rate"].max(), 50)
    ax.plot(xs, m * xs + b, color="black", linewidth=1.5, label=f"trend: y={m:.3f}x+{b:.2f}")
    ax.set_title("Attendance Rate vs GPA")
    ax.set_xlabel("Attendance rate (%)")
    ax.set_ylabel("GPA")
    ax.legend()
    _save(fig, "attendance_vs_gpa.png")


def plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, ax=ax, cbar_kws={"shrink": 0.8})
    else:
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Matrix (Numeric Features)")
    _save(fig, "correlation_heatmap.png")


def plot_pass_by_parent_education(df: pd.DataFrame) -> None:
    order = ["None", "High School", "Bachelor's", "Graduate"]
    crosstab = pd.crosstab(df["parent_education_level"], df["passed"]).reindex(order)
    # Guarantee both Yes/No columns exist so the chart renders even if the
    # cohort has no failures (or no passes).
    for col in ["Yes", "No"]:
        if col not in crosstab.columns:
            crosstab[col] = 0
    fig, ax = plt.subplots(figsize=(8, 5))
    crosstab[["Yes", "No"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=["#55A868", "#C44E52"], edgecolor="white",
    )
    ax.set_title("Pass / Fail by Parent Education Level")
    ax.set_xlabel("Parent education level")
    ax.set_ylabel("Number of students")
    ax.legend(title="Passed")
    plt.xticks(rotation=0)
    _save(fig, "pass_by_parent_education.png")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    total = len(df)
    avg_gpa = df["gpa"].mean()
    pass_rate = (df["passed"] == "Yes").mean() * 100
    top = top_gpa_correlations(corr, k=3)

    print()
    print("=" * 60)
    print("STUDENT PERFORMANCE — SUMMARY")
    print("=" * 60)
    print(f"Total students analyzed : {total}")
    print(f"Average GPA             : {avg_gpa:.2f}")
    print(f"Pass rate (GPA ≥ {PASS_THRESHOLD}) : {pass_rate:.1f}%")
    print()
    print("Top features correlated with GPA:")
    for feature, value in top:
        print(f"  {feature:<25} r = {value:+.3f}")
    print()
    print("Mean GPA by grade level:")
    print(group_stats(df, "grade_level"))
    print()
    print("Mean GPA by parent education:")
    print(group_stats(df, "parent_education_level"))
    print()
    print(f"Charts written to: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    print("Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} rows from {DATA_FILE}")

    print("\nCleaning data...")
    df = clean_data(df)
    print(f"  Final row count: {len(df)}")

    print("\nComputing correlations...")
    corr = correlation_matrix(df)

    print("\nGenerating charts...")
    plot_gpa_distribution(df)
    plot_avg_gpa_by_grade(df)
    plot_study_vs_gpa(df)
    plot_attendance_vs_gpa(df)
    plot_correlation_heatmap(corr)
    plot_pass_by_parent_education(df)

    print_summary(df, corr)


if __name__ == "__main__":
    run_pipeline()
