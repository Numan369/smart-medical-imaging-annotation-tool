from collections import Counter
from pathlib import Path
import csv
import random


INDEX_PATH = Path("prepared_data") / "dataset_index.csv"
OUTPUT_PATH = Path("prepared_data") / "dataset_splits.csv"

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10


if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"Dataset index was not found: {INDEX_PATH.resolve()}"
    )


positive_rows = []
negative_rows = []


# Read and validate the dataset index
with INDEX_PATH.open(
    "r",
    encoding="utf-8",
    newline="",
) as index_file:

    reader = csv.DictReader(index_file)

    expected_columns = {
        "ImageId",
        "DICOMPath",
        "HasPneumothorax",
        "MaskCount",
    }

    if set(reader.fieldnames or []) != expected_columns:
        raise ValueError(
            f"Unexpected index columns: {reader.fieldnames}"
        )

    for row_number, row in enumerate(reader, start=2):
        label = int(row["HasPneumothorax"])

        if label not in (0, 1):
            raise ValueError(
                f"Invalid label on row {row_number}: {label}"
            )

        if label == 1:
            positive_rows.append(row)
        else:
            negative_rows.append(row)


def split_class(rows, random_seed):
    """Split one label group into train, validation, and test sets."""

    shuffled_rows = rows.copy()
    random.Random(random_seed).shuffle(shuffled_rows)

    test_count = round(len(shuffled_rows) * TEST_FRACTION)
    validation_count = round(
        len(shuffled_rows) * VALIDATION_FRACTION
    )

    test_rows = shuffled_rows[:test_count]

    validation_rows = shuffled_rows[
        test_count:test_count + validation_count
    ]

    training_rows = shuffled_rows[
        test_count + validation_count:
    ]

    return training_rows, validation_rows, test_rows


# Split positive and negative images separately
positive_train, positive_validation, positive_test = split_class(
    positive_rows,
    RANDOM_SEED,
)

negative_train, negative_validation, negative_test = split_class(
    negative_rows,
    RANDOM_SEED + 1,
)


# Combine the two classes within each split
split_rows = {
    "train": positive_train + negative_train,
    "validation": positive_validation + negative_validation,
    "test": positive_test + negative_test,
}


# Shuffle each completed split reproducibly
for offset, rows in enumerate(split_rows.values()):
    random.Random(RANDOM_SEED + 10 + offset).shuffle(rows)


# Confirm that every image occurs exactly once
all_split_ids = [
    row["ImageId"]
    for rows in split_rows.values()
    for row in rows
]

if len(all_split_ids) != len(set(all_split_ids)):
    raise ValueError(
        "At least one image appears in more than one split."
    )

expected_total = len(positive_rows) + len(negative_rows)

if len(all_split_ids) != expected_total:
    raise ValueError(
        "The splits do not contain every indexed image."
    )


# Save one row per image with its assigned split
fieldnames = [
    "ImageId",
    "DICOMPath",
    "HasPneumothorax",
    "MaskCount",
    "Split",
]

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as output_file:

    writer = csv.DictWriter(
        output_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for split_name in ("train", "validation", "test"):
        for row in split_rows[split_name]:
            writer.writerow(
                {
                    **row,
                    "Split": split_name,
                }
            )


print("Dataset splits created successfully")
print("-----------------------------------")
print(f"Output file: {OUTPUT_PATH.resolve()}")
print(f"Random seed: {RANDOM_SEED}")

for split_name in ("train", "validation", "test"):
    rows = split_rows[split_name]

    label_counts = Counter(
        int(row["HasPneumothorax"])
        for row in rows
    )

    total = len(rows)
    positive_count = label_counts[1]
    negative_count = label_counts[0]
    positive_percentage = positive_count / total * 100

    print(f"\n{split_name.capitalize()} split")
    print(f"  Total images: {total}")
    print(f"  Positive images: {positive_count}")
    print(f"  Negative images: {negative_count}")
    print(
        f"  Positive percentage: "
        f"{positive_percentage:.2f}%"
    )