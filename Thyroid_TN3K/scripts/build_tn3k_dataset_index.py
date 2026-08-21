from pathlib import Path
import csv
import json

import numpy as np
from PIL import Image


# ============================================================
# PATHS
# ============================================================

THYROID_ROOT = Path(__file__).resolve().parent.parent

TN3K_ROOT = (
    THYROID_ROOT
    / "dataset"
    / "Thyroid Dataset"
    / "tn3k"
)

TRAINVAL_IMAGE_DIR = TN3K_ROOT / "trainval-image"
TRAINVAL_MASK_DIR = TN3K_ROOT / "trainval-mask"

TEST_IMAGE_DIR = TN3K_ROOT / "test-image"
TEST_MASK_DIR = TN3K_ROOT / "test-mask"

FOLD0_PATH = TN3K_ROOT / "tn3k-trainval-fold0.json"

OUTPUT_DIR = THYROID_ROOT / "prepared_data"

OUTPUT_CSV = OUTPUT_DIR / "tn3k_dataset_index.csv"


# ============================================================
# SETTINGS
# ============================================================

# TN3K masks are JPEG files.
#
# Visual inspection showed:
#
# background ≈ 0-11
# foreground ≈ 243-255
#
# Therefore >= 128 safely separates the two groups.
MASK_THRESHOLD = 128


# ============================================================
# HELPERS
# ============================================================

def load_fold0():

    with open(
        FOLD0_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    train_ids = {
        int(index)
        for index in data["train"]
    }

    val_ids = {
        int(index)
        for index in data["val"]
    }

    return train_ids, val_ids


def load_binary_mask(mask_path):

    with Image.open(mask_path) as mask_image:

        mask = np.array(
            mask_image.convert("L")
        )

    binary_mask = mask >= MASK_THRESHOLD

    return binary_mask


def calculate_mask_statistics(mask_path):

    binary_mask = load_binary_mask(
        mask_path
    )

    total_pixels = int(
        binary_mask.size
    )

    nodule_pixels = int(
        binary_mask.sum()
    )

    if total_pixels == 0:

        area_fraction = 0.0

    else:

        area_fraction = (
            nodule_pixels
            / total_pixels
        )

    return (
        nodule_pixels,
        total_pixels,
        area_fraction,
    )


def get_size_group(area_fraction):

    if area_fraction < 0.005:
        return "tiny"

    elif area_fraction < 0.02:
        return "small"

    elif area_fraction < 0.10:
        return "medium"

    else:
        return "large"


def relative_to_thyroid_root(path):

    return path.relative_to(
        THYROID_ROOT
    ).as_posix()


# ============================================================
# BUILD DEVELOPMENT ROWS
# ============================================================

def build_trainval_rows(
    train_ids,
    val_ids,
):

    rows = []

    image_files = sorted(
        TRAINVAL_IMAGE_DIR.glob("*.jpg")
    )

    print()
    print(
        f"Processing development images: "
        f"{len(image_files)}"
    )

    for counter, image_path in enumerate(
        image_files,
        start=1,
    ):

        original_id = image_path.stem

        index = int(original_id)

        mask_path = (
            TRAINVAL_MASK_DIR
            / image_path.name
        )

        if not mask_path.exists():

            raise FileNotFoundError(
                f"Missing mask:\n{mask_path}"
            )

        # ----------------------------------------------------
        # Determine Fold 0 split
        # ----------------------------------------------------

        if index in train_ids:

            split = "train"

        elif index in val_ids:

            split = "validation"

        else:

            raise ValueError(
                f"Image index {index} is not "
                f"present in Fold 0 train or val."
            )

        # ----------------------------------------------------
        # Image dimensions
        # ----------------------------------------------------

        with Image.open(image_path) as image:

            width, height = image.size

        # ----------------------------------------------------
        # Mask statistics
        # ----------------------------------------------------

        (
            nodule_pixels,
            total_pixels,
            area_fraction,
        ) = calculate_mask_statistics(
            mask_path
        )

        size_group = get_size_group(
            area_fraction
        )

        sample_id = (
            f"trainval_{original_id}"
        )

        rows.append(
            {
                "SampleId": sample_id,
                "OriginalId": original_id,
                "Split": split,
                "ImagePath": relative_to_thyroid_root(
                    image_path
                ),
                "MaskPath": relative_to_thyroid_root(
                    mask_path
                ),
                "Width": width,
                "Height": height,
                "NodulePixels": nodule_pixels,
                "TotalPixels": total_pixels,
                "NoduleAreaFraction": (
                    f"{area_fraction:.8f}"
                ),
                "NoduleAreaPercent": (
                    f"{area_fraction * 100:.4f}"
                ),
                "NoduleSizeGroup": size_group,
            }
        )

        if counter % 500 == 0:

            print(
                f"  Processed "
                f"{counter}/"
                f"{len(image_files)}"
            )

    return rows


# ============================================================
# BUILD OFFICIAL TEST ROWS
# ============================================================

def build_test_rows():

    rows = []

    image_files = sorted(
        TEST_IMAGE_DIR.glob("*.jpg")
    )

    print()
    print(
        f"Processing official test images: "
        f"{len(image_files)}"
    )

    for counter, image_path in enumerate(
        image_files,
        start=1,
    ):

        original_id = image_path.stem

        mask_path = (
            TEST_MASK_DIR
            / image_path.name
        )

        if not mask_path.exists():

            raise FileNotFoundError(
                f"Missing test mask:\n"
                f"{mask_path}"
            )

        with Image.open(image_path) as image:

            width, height = image.size

        (
            nodule_pixels,
            total_pixels,
            area_fraction,
        ) = calculate_mask_statistics(
            mask_path
        )

        size_group = get_size_group(
            area_fraction
        )

        sample_id = (
            f"test_{original_id}"
        )

        rows.append(
            {
                "SampleId": sample_id,
                "OriginalId": original_id,
                "Split": "test",
                "ImagePath": relative_to_thyroid_root(
                    image_path
                ),
                "MaskPath": relative_to_thyroid_root(
                    mask_path
                ),
                "Width": width,
                "Height": height,
                "NodulePixels": nodule_pixels,
                "TotalPixels": total_pixels,
                "NoduleAreaFraction": (
                    f"{area_fraction:.8f}"
                ),
                "NoduleAreaPercent": (
                    f"{area_fraction * 100:.4f}"
                ),
                "NoduleSizeGroup": size_group,
            }
        )

        if counter % 200 == 0:

            print(
                f"  Processed "
                f"{counter}/"
                f"{len(image_files)}"
            )

    return rows


# ============================================================
# VALIDATE INDEX
# ============================================================

def validate_rows(rows):

    print()
    print("=" * 70)
    print("VALIDATING DATASET INDEX")
    print("=" * 70)

    sample_ids = [
        row["SampleId"]
        for row in rows
    ]

    unique_sample_ids = set(
        sample_ids
    )

    if len(sample_ids) != len(
        unique_sample_ids
    ):

        raise ValueError(
            "Duplicate SampleId found."
        )

    train_count = sum(
        row["Split"] == "train"
        for row in rows
    )

    validation_count = sum(
        row["Split"] == "validation"
        for row in rows
    )

    test_count = sum(
        row["Split"] == "test"
        for row in rows
    )

    print(
        f"Train rows:      "
        f"{train_count}"
    )

    print(
        f"Validation rows: "
        f"{validation_count}"
    )

    print(
        f"Test rows:       "
        f"{test_count}"
    )

    print(
        f"Total rows:      "
        f"{len(rows)}"
    )

    print(
        f"Unique SampleIds:"
        f" {len(unique_sample_ids)}"
    )

    if train_count != 2303:

        raise ValueError(
            "Unexpected Fold 0 "
            "training count."
        )

    if validation_count != 576:

        raise ValueError(
            "Unexpected Fold 0 "
            "validation count."
        )

    if test_count != 614:

        raise ValueError(
            "Unexpected official "
            "test count."
        )

    if len(rows) != 3493:

        raise ValueError(
            "Unexpected total "
            "dataset count."
        )


# ============================================================
# WRITE CSV
# ============================================================

def write_csv(rows):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "SampleId",
        "OriginalId",
        "Split",
        "ImagePath",
        "MaskPath",
        "Width",
        "Height",
        "NodulePixels",
        "TotalPixels",
        "NoduleAreaFraction",
        "NoduleAreaPercent",
        "NoduleSizeGroup",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

def print_summary(rows):

    print()
    print("=" * 70)
    print("TN3K DATASET INDEX SUMMARY")
    print("=" * 70)

    for split in [
        "train",
        "validation",
        "test",
    ]:

        split_rows = [
            row
            for row in rows
            if row["Split"] == split
        ]

        print()
        print(split.upper())

        print(
            f"  Images: "
            f"{len(split_rows)}"
        )

        groups = {
            "tiny": 0,
            "small": 0,
            "medium": 0,
            "large": 0,
        }

        for row in split_rows:

            groups[
                row["NoduleSizeGroup"]
            ] += 1

        for group, count in groups.items():

            print(
                f"  {group:<6}: "
                f"{count}"
            )

    print()
    print(f"Output CSV:")
    print(OUTPUT_CSV)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING TN3K DATASET INDEX")
    print("=" * 70)

    print()
    print("Using development fold:")
    print("Fold 0")

    print()
    print(
        f"Ground-truth mask threshold: "
        f">= {MASK_THRESHOLD}"
    )

    # --------------------------------------------------------
    # Fold 0
    # --------------------------------------------------------

    train_ids, val_ids = (
        load_fold0()
    )

    print()
    print(
        f"Fold 0 training IDs:   "
        f"{len(train_ids)}"
    )

    print(
        f"Fold 0 validation IDs: "
        f"{len(val_ids)}"
    )

    # --------------------------------------------------------
    # Build rows
    # --------------------------------------------------------

    trainval_rows = (
        build_trainval_rows(
            train_ids,
            val_ids,
        )
    )

    test_rows = (
        build_test_rows()
    )

    rows = (
        trainval_rows
        + test_rows
    )

    # --------------------------------------------------------
    # Validate and save
    # --------------------------------------------------------

    validate_rows(rows)

    write_csv(rows)

    print_summary(rows)

    print()
    print("=" * 70)
    print("DATASET INDEX CREATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()