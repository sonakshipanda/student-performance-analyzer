"""
Synthetic student performance dataset generator.

Produces a CSV of 500+ student records with realistic correlations between
study habits, attendance, socioeconomic factors, and academic outcomes.
Missing values and noise are intentionally injected so the cleaning step
in main.py has real work to do.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)

N_STUDENTS = 550
OUTPUT_CSV = "students_raw.csv"

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Sam", "Avery",
    "Quinn", "Reese", "Cameron", "Drew", "Hayden", "Parker", "Sage", "Rowan",
    "Emery", "Finley", "Skyler", "Dakota", "Jamie", "Kai", "Logan", "Micah",
]
LAST_NAMES = [
    "Smith", "Johnson", "Lee", "Garcia", "Patel", "Nguyen", "Brown", "Wilson",
    "Kim", "Martinez", "Davis", "Lopez", "Singh", "Chen", "Rodriguez", "Khan",
    "Anderson", "Thomas", "Moore", "Jackson", "White", "Harris", "Clark",
]


def _sample_parent_education(n: int) -> np.ndarray:
    levels = ["None", "High School", "Bachelor's", "Graduate"]
    probs = [0.08, 0.42, 0.35, 0.15]
    return RNG.choice(levels, size=n, p=probs)


def _education_boost(parent_ed: np.ndarray) -> np.ndarray:
    """Small positive boost to baseline GPA based on parent education level."""
    mapping = {"None": -0.15, "High School": 0.0, "Bachelor's": 0.15, "Graduate": 0.25}
    return np.array([mapping[p] for p in parent_ed])


def _generate_names(n: int) -> list[str]:
    firsts = RNG.choice(FIRST_NAMES, size=n)
    lasts = RNG.choice(LAST_NAMES, size=n)
    return [f"{f} {l}" for f, l in zip(firsts, lasts)]


def generate_dataset(n: int = N_STUDENTS) -> pd.DataFrame:
    """Generate a synthetic student dataset with realistic correlations."""
    student_id = np.arange(1000, 1000 + n)
    name = _generate_names(n)
    grade_level = RNG.choice([9, 10, 11, 12], size=n, p=[0.27, 0.26, 0.24, 0.23])
    gender = RNG.choice(["Male", "Female", "Non-binary"], size=n, p=[0.48, 0.48, 0.04])

    attendance_rate = np.clip(RNG.normal(loc=88, scale=8, size=n), 50, 100)
    study_hours = np.clip(RNG.normal(loc=8, scale=4, size=n), 0, 20).astype(int)

    parent_education = _sample_parent_education(n)
    ed_boost = _education_boost(parent_education)

    free_lunch_prob = np.where(
        parent_education == "None", 0.70,
        np.where(parent_education == "High School", 0.45,
                 np.where(parent_education == "Bachelor's", 0.20, 0.10))
    )
    free_lunch = RNG.binomial(1, free_lunch_prob, size=n)

    internet_access = RNG.binomial(1, np.where(free_lunch == 1, 0.75, 0.95), size=n)
    extracurricular = RNG.binomial(1, 0.55, size=n)

    # Latent academic ability — z-score-like, mean ~0, sd ~1.
    ability = (
        ((study_hours - 8) / 4) * 0.45
        + ((attendance_rate - 88) / 8) * 0.45
        + ed_boost * 1.5
        + (extracurricular - 0.5) * 0.15
        + (internet_access - 0.85) * 0.30
        + RNG.normal(0, 0.55, size=n)
    )

    # Derive each subject score from ability. Baseline 70 with ±~15 spread
    # produces a realistic mix where ~10–20% fall below the pass threshold.
    def score_from_ability(noise_scale: float, offset: float = 0.0) -> np.ndarray:
        raw = 70 + ability * 12 + offset + RNG.normal(0, noise_scale, size=n)
        return np.clip(raw, 0, 100).round().astype(int)

    math_score = score_from_ability(6, offset=-2)
    science_score = score_from_ability(6, offset=0)
    english_score = score_from_ability(5, offset=2)
    history_score = score_from_ability(7, offset=0)

    avg_score = (math_score + science_score + english_score + history_score) / 4
    # Map average score onto a 0-4 GPA scale; 60 ≈ 2.0 (pass cutoff).
    gpa = np.clip(((avg_score - 30) / 17.5) + RNG.normal(0, 0.10, size=n), 0.0, 4.0).round(2)

    passed = np.where(gpa >= 2.0, "Yes", "No")

    df = pd.DataFrame({
        "student_id": student_id,
        "name": name,
        "grade_level": grade_level,
        "gender": gender,
        "attendance_rate": attendance_rate.round(1),
        "study_hours_per_week": study_hours,
        "math_score": math_score,
        "science_score": science_score,
        "english_score": english_score,
        "history_score": history_score,
        "gpa": gpa,
        "extracurricular_activities": np.where(extracurricular == 1, "Yes", "No"),
        "internet_access": np.where(internet_access == 1, "Yes", "No"),
        "parent_education_level": parent_education,
        "free_lunch": np.where(free_lunch == 1, "Yes", "No"),
        "passed": passed,
    })

    df = _inject_noise(df)
    return df


def _inject_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Inject missing values, duplicates, and a few dirty types for cleaning practice."""
    df = df.copy()
    n = len(df)

    # Missing values across several columns at varying rates.
    null_targets = {
        "attendance_rate": 0.04,
        "study_hours_per_week": 0.03,
        "parent_education_level": 0.05,
        "internet_access": 0.02,
        "english_score": 0.02,
        "name": 0.03,
    }
    for col, rate in null_targets.items():
        mask = RNG.random(n) < rate
        df.loc[mask, col] = np.nan

    # A handful of attendance values written as strings with a % sign.
    df["attendance_rate"] = df["attendance_rate"].astype(object)
    str_idx = RNG.choice(df.index, size=8, replace=False)
    df.loc[str_idx, "attendance_rate"] = df.loc[str_idx, "attendance_rate"].apply(
        lambda v: f"{v}%" if pd.notna(v) else v
    )

    # Duplicate a few rows verbatim.
    dup_rows = df.sample(n=6, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def main() -> None:
    df = generate_dataset()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
