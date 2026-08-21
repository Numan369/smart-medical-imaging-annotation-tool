from pathlib import Path
from collections import Counter
import json


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

IMAGE_DIR = TN3K_ROOT / "trainval-image"
MASK_DIR = TN3K_ROOT / "trainval-mask"


# ============================================================
# HELPERS
# ============================================================

def load_fold(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def index_to_filename(index):
    """
    TN3K development images use zero-padded names:
    0 -> 0000.jpg
    1 -> 0001.jpg
    ...
    """
    return f"{int(index):04d}.jpg"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TN3K OFFICIAL FOLD VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Actual development files
    # --------------------------------------------------------

    image_files = sorted(IMAGE_DIR.glob("*.jpg"))
    mask_files = sorted(MASK_DIR.glob("*.jpg"))

    image_names = {path.name for path in image_files}
    mask_names = {path.name for path in mask_files}

    print()
    print(f"Development images: {len(image_names)}")
    print(f"Development masks:  {len(mask_names)}")

    # --------------------------------------------------------
    # Fold files
    # --------------------------------------------------------

    fold_files = sorted(
        TN3K_ROOT.glob("tn3k-trainval-fold*.json")
    )

    print(f"Fold files:         {len(fold_files)}")

    all_validation_indices = []

    print()

    # --------------------------------------------------------
    # Verify every fold
    # --------------------------------------------------------

    for fold_number, fold_path in enumerate(fold_files):

        data = load_fold(fold_path)

        train = [int(x) for x in data["train"]]
        val = [int(x) for x in data["val"]]

        train_set = set(train)
        val_set = set(val)

        overlap = train_set & val_set
        union = train_set | val_set

        train_duplicates = len(train) - len(train_set)
        val_duplicates = len(val) - len(val_set)

        # Convert JSON indices into expected filenames
        train_names = {
            index_to_filename(index)
            for index in train
        }

        val_names = {
            index_to_filename(index)
            for index in val
        }

        expected_names = train_names | val_names

        missing_images = sorted(
            expected_names - image_names
        )

        missing_masks = sorted(
            expected_names - mask_names
        )

        extra_images = sorted(
            image_names - expected_names
        )

        print("-" * 70)
        print(f"FOLD {fold_number}")
        print("-" * 70)

        print(f"Train entries:          {len(train)}")
        print(f"Validation entries:     {len(val)}")
        print(f"Train duplicates:       {train_duplicates}")
        print(f"Validation duplicates:  {val_duplicates}")
        print(f"Train/val overlap:      {len(overlap)}")
        print(f"Unique total indices:   {len(union)}")

        print(f"Missing image files:    {len(missing_images)}")
        print(f"Missing mask files:     {len(missing_masks)}")
        print(f"Unreferenced images:    {len(extra_images)}")

        if overlap:
            print(
                "First overlapping indices:",
                sorted(overlap)[:10],
            )

        if missing_images:
            print(
                "First missing images:",
                missing_images[:10],
            )

        if missing_masks:
            print(
                "First missing masks:",
                missing_masks[:10],
            )

        all_validation_indices.extend(val)

        print()

    # --------------------------------------------------------
    # Cross-fold validation coverage
    # --------------------------------------------------------

    print("=" * 70)
    print("CROSS-FOLD VALIDATION COVERAGE")
    print("=" * 70)

    counter = Counter(all_validation_indices)

    unique_validation = set(
        all_validation_indices
    )

    repeated_validation = sorted(
        index
        for index, count in counter.items()
        if count > 1
    )

    never_validation = sorted(
        set(range(len(image_names)))
        - unique_validation
    )

    print()
    print(
        f"Total validation entries across folds: "
        f"{len(all_validation_indices)}"
    )

    print(
        f"Unique validation indices:             "
        f"{len(unique_validation)}"
    )

    print(
        f"Indices repeated as validation:        "
        f"{len(repeated_validation)}"
    )

    print(
        f"Indices never used as validation:      "
        f"{len(never_validation)}"
    )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    print()
    print("=" * 70)

    valid = (
        len(image_names) == 2879
        and len(mask_names) == 2879
        and len(unique_validation) == 2879
        and len(repeated_validation) == 0
        and len(never_validation) == 0
    )

    if valid:

        print("RESULT: TN3K FOLD STRUCTURE VERIFIED")

    else:

        print("RESULT: FOLD STRUCTURE NEEDS INVESTIGATION")

    print("=" * 70)


if __name__ == "__main__":
    main()