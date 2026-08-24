"""Download the Trillion Dollar Words FOMC splits (not vendored in git)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

GITHUB_RAW = (
    "https://raw.githubusercontent.com/gtfintechlab/fomc-hawkish-dovish/"
    "main/training_data/test-and-training"
)
# Combined-S in the paper (sentence-split). Combined (no split) is lab-manual-combine.
DEFAULT_SEED = "5768"
DEFAULT_SPLIT = "lab-manual-split-combine"
DATA_DIR = Path(__file__).resolve().parent / "data"

LABEL_NAMES = {0: "dovish", 1: "hawkish", 2: "neutral"}

# Spreadsheet column names vary slightly across files; try these in order.
TEXT_CANDIDATES = ("sentence", "text", "Sentence", "Text")
LABEL_CANDIDATES = ("label", "Label", "labels", "class", "target")


def _url(kind: str, seed: str = DEFAULT_SEED, split: str = DEFAULT_SPLIT) -> str:
    folder = "training_data" if kind == "train" else "test_data"
    return f"{GITHUB_RAW}/{folder}/{split}-{kind}-{seed}.xlsx"


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(
        f"Could not find a {role} column. Available: {list(df.columns)}"
    )


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _pick_column(df, TEXT_CANDIDATES, "text")
    label_col = _pick_column(df, LABEL_CANDIDATES, "label")
    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": pd.to_numeric(df[label_col], errors="coerce").astype("Int64"),
        }
    )
    out = out.dropna(subset=["text", "label"]).copy()
    out["label"] = out["label"].astype(int)
    out["label_name"] = out["label"].map(LABEL_NAMES)
    return out.reset_index(drop=True)


def download_split(
    kind: str,
    seed: str = DEFAULT_SEED,
    split: str = DEFAULT_SPLIT,
    data_dir: Path | None = None,
) -> Path:
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / f"{split}-{kind}-{seed}.xlsx"
    if dest.exists():
        return dest
    url = _url(kind, seed, split)
    print(f"Downloading {url}")
    df = pd.read_excel(url)
    df.to_excel(dest, index=False)
    return dest


def load_splits(
    seed: str = DEFAULT_SEED,
    split: str = DEFAULT_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = download_split("train", str(seed), split)
    test_path = download_split("test", str(seed), split)
    train = normalize(pd.read_excel(train_path))
    test = normalize(pd.read_excel(test_path))
    return train, test


def download_paper_splits(
    split: str = DEFAULT_SPLIT,
    seeds: tuple[int, ...] | list[int] | None = None,
) -> list[Path]:
    """Download Combined-S train/test xlsx for the paper's three seeds."""
    if seeds is None:
        seeds = (5768, 78516, 944601)
    paths: list[Path] = []
    for seed in seeds:
        paths.append(download_split("train", str(seed), split))
        paths.append(download_split("test", str(seed), split))
    return paths


def train_val_test(
    seed: str | int = DEFAULT_SEED,
    split: str = DEFAULT_SPLIT,
    val_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Paper split: official train/test xlsx, then 20% of train held out as val."""
    import numpy as np

    train, test = load_splits(seed=str(seed), split=split)
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(len(train))
    n_val = int(len(train) * val_fraction)
    val = train.iloc[perm[:n_val]].reset_index(drop=True)
    train = train.iloc[perm[n_val:]].reset_index(drop=True)
    return train, val, test


if __name__ == "__main__":
    train, test = load_splits()
    print(f"train={len(train):,}  test={len(test):,}")
    print(train["label_name"].value_counts().to_string())
    print(train.head(3))
