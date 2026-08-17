"""
Create the fixed 45% positive / 55% negative dataset for V4B.

This script does not copy or modify DICOM files. It creates CSV manifests
containing a fixed set of image IDs selected from the existing train,
validation and test splits.

Expected source files:
    prepared_data/dataset_splits.csv
    prepared_data/dataset_index.csv  (used only if labels are not in splits)

Created files:
    prepared_data/v4b_fixed_45_55/v4b_fixed_45_55_splits.csv
    prepared_data/v4b_fixed_45_55/v4b_fixed_train.csv
    prepared_data/v4b_fixed_45_55/v4b_fixed_validation.csv
    prepared_data/v4b_fixed_45_55/v4b_fixed_test.csv
    prepared_data/v4b_fixed_45_55/v4b_fixed_45_55_summary.json
"""

import json
import re
from pathlib import Path

import pandas as pd


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PREPARED_DATA_DIRECTORY = PROJECT_ROOT / "prepared_data"

SOURCE_SPLITS_PATH = (
    PREPARED_DATA_DIRECTORY / "dataset_splits.csv"
)

SOURCE_INDEX_PATH = (
    PREPARED_DATA_DIRECTORY / "dataset_index.csv"
)

OUTPUT_DIRECTORY = (
    PREPARED_DATA_DIRECTORY / "v4b_fixed_45_55"
)

OUTPUT_MANIFEST_PATH = (
    OUTPUT_DIRECTORY / "v4b_fixed_45_55_splits.csv"
)

OUTPUT_SUMMARY_PATH = (
    OUTPUT_DIRECTORY / "v4b_fixed_45_55_summary.json"
)

RANDOM_SEED = 42

POSITIVE_PERCENT = 45
NEGATIVE_PERCENT = 55


# =========================================================
# Column detection
# =========================================================

def normalize_column_name(name):
    """Convert a column name to a simple comparable form."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(dataframe, candidates, required=True):
    """
    Find a dataframe column using several possible names.
    """

    normalized_columns = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)

        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]

    if required:
        raise KeyError(
            "Could not find a required column.\n"
            f"Expected one of: {candidates}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


# =========================================================
# Value conversion
# =========================================================

def convert_label(value):
    """
    Convert common positive/negative label formats into 0 or 1.
    """

    if pd.isna(value):
        raise ValueError("A missing label value was found.")

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return 1 if float(value) > 0 else 0

    text = str(value).strip().lower()

    positive_values = {
        "1",
        "true",
        "yes",
        "positive",
        "pneumothorax",
        "present",
    }

    negative_values = {
        "0",
        "false",
        "no",
        "negative",
        "normal",
        "absent",
        "-1",
    }

    if text in positive_values:
        return 1

    if text in negative_values:
        return 0

    try:
        return 1 if float(text) > 0 else 0
    except ValueError as error:
        raise ValueError(
            f"Could not convert label value to 0 or 1: {value!r}"
        ) from error


def convert_split(value):
    """Convert split names into train, validation or test."""

    text = str(value).strip().lower()

    if text in {"train", "training"}:
        return "train"

    if text in {"val", "valid", "validation", "validate"}:
        return "validation"

    if text in {"test", "testing"}:
        return "test"

    raise ValueError(f"Unknown dataset split value: {value!r}")


# =========================================================
# Load the source dataset
# =========================================================

def load_source_dataset():
    if not SOURCE_SPLITS_PATH.exists():
        raise FileNotFoundError(
            "The source split file was not found:\n"
            f"{SOURCE_SPLITS_PATH}"
        )

    print(f"Reading: {SOURCE_SPLITS_PATH}")

    dataframe = pd.read_csv(SOURCE_SPLITS_PATH)

    print(f"Source rows: {len(dataframe):,}")
    print(f"Source columns: {list(dataframe.columns)}")

    image_id_column = find_column(
        dataframe,
        [
            "ImageId",
            "image_id",
            "imageid",
            "id",
            "dicom_id",
            "patient_id",
        ],
    )

    split_column = find_column(
        dataframe,
        [
            "split",
            "dataset_split",
            "set",
            "subset",
        ],
    )

    label_column = find_column(
        dataframe,
        [
            "label",
            "is_positive",
            "positive",
            "has_mask",
            "has_pneumothorax",
            "pneumothorax",
            "target",
            "class",
            "mask_count",
        ],
        required=False,
    )

    dataframe = dataframe.copy()

    dataframe["__v4b_image_id"] = (
        dataframe[image_id_column].astype(str).str.strip()
    )

    # If the split file does not contain labels, obtain them
    # from dataset_index.csv.
    if label_column is None:
        if not SOURCE_INDEX_PATH.exists():
            raise FileNotFoundError(
                "The split CSV has no recognizable label column, and "
                "dataset_index.csv was not found:\n"
                f"{SOURCE_INDEX_PATH}"
            )

        print(
            "No label column found in dataset_splits.csv.\n"
            f"Reading labels from: {SOURCE_INDEX_PATH}"
        )

        index_dataframe = pd.read_csv(SOURCE_INDEX_PATH)

        index_id_column = find_column(
            index_dataframe,
            [
                "ImageId",
                "image_id",
                "imageid",
                "id",
                "dicom_id",
                "patient_id",
            ],
        )

        index_label_column = find_column(
            index_dataframe,
            [
                "label",
                "is_positive",
                "positive",
                "has_mask",
                "has_pneumothorax",
                "pneumothorax",
                "target",
                "class",
                "mask_count",
            ],
        )

        label_lookup = index_dataframe[
            [index_id_column, index_label_column]
        ].copy()

        label_lookup["__v4b_image_id"] = (
            label_lookup[index_id_column]
            .astype(str)
            .str.strip()
        )

        label_lookup = label_lookup[
            ["__v4b_image_id", index_label_column]
        ].rename(
            columns={
                index_label_column: "__v4b_source_label"
            }
        )

        if label_lookup["__v4b_image_id"].duplicated().any():
            duplicate_count = int(
                label_lookup["__v4b_image_id"]
                .duplicated()
                .sum()
            )

            raise ValueError(
                f"dataset_index.csv contains {duplicate_count} "
                "duplicate image IDs."
            )

        dataframe = dataframe.merge(
            label_lookup,
            on="__v4b_image_id",
            how="left",
            validate="one_to_one",
        )

        if dataframe["__v4b_source_label"].isna().any():
            missing_count = int(
                dataframe["__v4b_source_label"].isna().sum()
            )

            raise ValueError(
                f"{missing_count} split rows could not be matched "
                "with labels from dataset_index.csv."
            )

        source_labels = dataframe["__v4b_source_label"]

    else:
        print(f"Detected label column: {label_column}")
        source_labels = dataframe[label_column]

    print(f"Detected image ID column: {image_id_column}")
    print(f"Detected split column: {split_column}")

    dataframe["v4b_label"] = source_labels.map(convert_label)
    dataframe["v4b_split"] = dataframe[split_column].map(
        convert_split
    )

    if dataframe["__v4b_image_id"].duplicated().any():
        duplicates = dataframe.loc[
            dataframe["__v4b_image_id"].duplicated(False),
            "__v4b_image_id",
        ].tolist()

        raise ValueError(
            "Duplicate image IDs were found in the source splits.\n"
            f"Examples: {duplicates[:10]}"
        )

    return dataframe, image_id_column


# =========================================================
# Select the fixed 45/55 subsets
# =========================================================

def calculate_required_negatives(positive_count):
    return round(
        positive_count
        * NEGATIVE_PERCENT
        / POSITIVE_PERCENT
    )


def create_fixed_split(dataframe, split_name):
    split_dataframe = dataframe[
        dataframe["v4b_split"] == split_name
    ].copy()

    positive_rows = split_dataframe[
        split_dataframe["v4b_label"] == 1
    ].copy()

    negative_rows = split_dataframe[
        split_dataframe["v4b_label"] == 0
    ].copy()

    positive_count = len(positive_rows)
    available_negative_count = len(negative_rows)

    required_negative_count = calculate_required_negatives(
        positive_count
    )

    if positive_count == 0:
        raise ValueError(
            f"No positive images were found in the {split_name} split."
        )

    if available_negative_count < required_negative_count:
        raise ValueError(
            f"Not enough negative images in {split_name}.\n"
            f"Required: {required_negative_count}\n"
            f"Available: {available_negative_count}"
        )

    # Negatives are sampled exactly once.
    fixed_negative_rows = negative_rows.sample(
        n=required_negative_count,
        replace=False,
        random_state=RANDOM_SEED,
    )

    fixed_split = pd.concat(
        [positive_rows, fixed_negative_rows],
        ignore_index=True,
    )

    # Fixed deterministic row order.
    fixed_split = fixed_split.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    fixed_split["v4b_sampling"] = "fixed"
    fixed_split["v4b_selection_seed"] = RANDOM_SEED
    fixed_split["v4b_row_number"] = range(
        1,
        len(fixed_split) + 1,
    )

    return fixed_split


# =========================================================
# Validation
# =========================================================

def verify_no_split_leakage(fixed_splits):
    split_ids = {
        split_name: set(
            split_dataframe["__v4b_image_id"]
        )
        for split_name, split_dataframe in fixed_splits.items()
    }

    comparisons = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for first_split, second_split in comparisons:
        overlap = (
            split_ids[first_split]
            & split_ids[second_split]
        )

        if overlap:
            raise ValueError(
                "Dataset leakage detected between "
                f"{first_split} and {second_split}.\n"
                f"Overlapping image IDs: {list(overlap)[:10]}"
            )


def calculate_split_summary(dataframe):
    positive_count = int(
        (dataframe["v4b_label"] == 1).sum()
    )

    negative_count = int(
        (dataframe["v4b_label"] == 0).sum()
    )

    total_count = len(dataframe)

    return {
        "total": total_count,
        "positive": positive_count,
        "negative": negative_count,
        "positive_percent": (
            positive_count / total_count * 100
        ),
        "negative_percent": (
            negative_count / total_count * 100
        ),
    }


# =========================================================
# Main
# =========================================================

def main():
    print("=" * 65)
    print("V4B FIXED 45/55 DATASET CREATION")
    print("=" * 65)
    print(f"Random seed: {RANDOM_SEED}")
    print(
        f"Target ratio: {POSITIVE_PERCENT}% positive / "
        f"{NEGATIVE_PERCENT}% negative"
    )
    print()

    source_dataframe, image_id_column = (
        load_source_dataset()
    )

    fixed_splits = {
        "train": create_fixed_split(
            source_dataframe,
            "train",
        ),
        "validation": create_fixed_split(
            source_dataframe,
            "validation",
        ),
        "test": create_fixed_split(
            source_dataframe,
            "test",
        ),
    }

    verify_no_split_leakage(fixed_splits)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "experiment": "V4B fixed 45/55 dataset",
        "random_seed": RANDOM_SEED,
        "target_positive_percent": POSITIVE_PERCENT,
        "target_negative_percent": NEGATIVE_PERCENT,
        "image_id_column": image_id_column,
        "source_splits": str(SOURCE_SPLITS_PATH),
        "splits": {},
    }

    output_names = {
        "train": "v4b_fixed_train.csv",
        "validation": "v4b_fixed_validation.csv",
        "test": "v4b_fixed_test.csv",
    }

    for split_name, fixed_dataframe in fixed_splits.items():
        # Remove temporary internal columns.
        save_dataframe = fixed_dataframe.drop(
            columns=[
                "__v4b_source_label",
            ],
            errors="ignore",
        ).copy()

        output_path = (
            OUTPUT_DIRECTORY / output_names[split_name]
        )

        save_dataframe.to_csv(
            output_path,
            index=False,
        )

        split_summary = calculate_split_summary(
            fixed_dataframe
        )

        summary["splits"][split_name] = split_summary

        print()
        print(f"{split_name.upper()} SPLIT")
        print("-" * 35)
        print(f"Total:     {split_summary['total']:,}")
        print(f"Positive:  {split_summary['positive']:,}")
        print(f"Negative:  {split_summary['negative']:,}")
        print(
            "Ratio:     "
            f"{split_summary['positive_percent']:.2f}% positive / "
            f"{split_summary['negative_percent']:.2f}% negative"
        )
        print(f"Saved:     {output_path}")

    complete_manifest = pd.concat(
        [
            fixed_splits["train"],
            fixed_splits["validation"],
            fixed_splits["test"],
        ],
        ignore_index=True,
    )

    complete_manifest = complete_manifest.drop(
        columns=[
            "__v4b_source_label",
        ],
        errors="ignore",
    )

    complete_manifest.to_csv(
        OUTPUT_MANIFEST_PATH,
        index=False,
    )

    with OUTPUT_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary,
            summary_file,
            indent=4,
        )

    print()
    print("=" * 65)
    print("V4B FIXED DATASET CREATED SUCCESSFULLY")
    print("=" * 65)
    print("Duplicate image IDs: 0")
    print("Train/validation/test leakage: 0")
    print(f"Complete manifest: {OUTPUT_MANIFEST_PATH}")
    print(f"Summary: {OUTPUT_SUMMARY_PATH}")
    print()
    print(
        "The selected negative image identities are now fixed. "
        "Running this script again with seed 42 will select the "
        "same images."
    )


if __name__ == "__main__":
    main()