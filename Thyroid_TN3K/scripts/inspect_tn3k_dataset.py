from pathlib import Path
from collections import Counter
import json

import numpy as np
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

# This file is located at:
#
# Smart Medical Imaging Annotation Tool/
# └── Thyroid_TN3K/
#     └── scripts/
#         └── inspect_tn3k_dataset.py
#
# Therefore:
# script.parent           -> scripts
# script.parent.parent    -> Thyroid_TN3K

THYROID_ROOT = Path(__file__).resolve().parent.parent

TN3K_ROOT = (
    THYROID_ROOT
    / "dataset"
    / "Thyroid Dataset"
    / "tn3k"
)

TRAIN_IMAGE_DIR = TN3K_ROOT / "trainval-image"
TRAIN_MASK_DIR = TN3K_ROOT / "trainval-mask"

TEST_IMAGE_DIR = TN3K_ROOT / "test-image"
TEST_MASK_DIR = TN3K_ROOT / "test-mask"


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(folder):
    """
    Return all supported image files in a folder.
    """

    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def get_extension_counts(files):
    """
    Count how many files use each extension.
    """

    return Counter(
        path.suffix.lower()
        for path in files
    )


def pair_images_and_masks(image_files, mask_files):
    """
    Pair images and masks using filename stem.

    Example:

        image:
            123.png

        mask:
            123.png

    Both have stem:
            123
    """

    image_map = {
        path.stem: path
        for path in image_files
    }

    mask_map = {
        path.stem: path
        for path in mask_files
    }

    image_ids = set(image_map.keys())
    mask_ids = set(mask_map.keys())

    matched_ids = sorted(
        image_ids & mask_ids
    )

    missing_masks = sorted(
        image_ids - mask_ids
    )

    orphan_masks = sorted(
        mask_ids - image_ids
    )

    return (
        matched_ids,
        missing_masks,
        orphan_masks,
        image_map,
        mask_map,
    )


def inspect_mask(mask_path):
    """
    Read a mask and calculate how much of
    the image belongs to the thyroid nodule.
    """

    with Image.open(mask_path) as mask_image:
        mask_array = np.array(mask_image)

    # Some segmentation masks may be grayscale,
    # while others could theoretically contain
    # multiple channels.
    if mask_array.ndim == 3:

      foreground = np.any(
        mask_array[..., :3] >= 128,
        axis=-1,
    )

    else:

      foreground = mask_array >= 128

    total_pixels = foreground.size

    foreground_pixels = int(
        foreground.sum()
    )

    if total_pixels == 0:
        foreground_ratio = 0.0
    else:
        foreground_ratio = (
            foreground_pixels
            / total_pixels
        )

    return {
        "shape": mask_array.shape,
        "foreground_pixels": foreground_pixels,
        "total_pixels": total_pixels,
        "foreground_ratio": foreground_ratio,
    }


def get_resolution_counts(files):
    """
    Count image resolutions.
    """

    counts = Counter()

    for path in files:

        with Image.open(path) as image:
            counts[image.size] += 1

    return counts


# ============================================================
# INSPECT IMAGE / MASK PAIRS
# ============================================================

def inspect_dataset_split(
    split_name,
    matched_ids,
    image_map,
    mask_map,
):
    """
    Inspect one dataset split.

    Examples:
        TRAIN / VALIDATION
        TEST
    """

    print()
    print("=" * 70)
    print(split_name)
    print("=" * 70)

    size_mismatches = []
    empty_masks = []

    lesion_ratios = []

    for index, image_id in enumerate(
        matched_ids,
        start=1,
    ):

        image_path = image_map[image_id]
        mask_path = mask_map[image_id]

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        with Image.open(image_path) as image:

            image_size = image.size
            image_mode = image.mode

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        with Image.open(mask_path) as mask:

            mask_size = mask.size
            mask_mode = mask.mode

        if image_size != mask_size:

            size_mismatches.append(
                (
                    image_id,
                    image_size,
                    mask_size,
                )
            )

        mask_info = inspect_mask(mask_path)

        lesion_ratio = (
            mask_info["foreground_ratio"]
        )

        lesion_ratios.append(
            lesion_ratio
        )

        if (
            mask_info["foreground_pixels"]
            == 0
        ):

            empty_masks.append(
                image_id
            )

        # ----------------------------------------------------
        # Print first 5 examples
        # ----------------------------------------------------

        if index <= 5:

            print()
            print(f"Example {index}")
            print(f"  ID: {image_id}")
            print(
                f"  Image file: "
                f"{image_path.name}"
            )
            print(
                f"  Mask file:  "
                f"{mask_path.name}"
            )
            print(
                f"  Image size: "
                f"{image_size}"
            )
            print(
                f"  Mask size:  "
                f"{mask_size}"
            )
            print(
                f"  Image mode: "
                f"{image_mode}"
            )
            print(
                f"  Mask mode:  "
                f"{mask_mode}"
            )
            print(
                f"  Nodule area: "
                f"{lesion_ratio * 100:.4f}%"
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("PAIR CHECK RESULTS")
    print("-" * 70)

    print(
        f"Matched image-mask pairs: "
        f"{len(matched_ids)}"
    )

    print(
        f"Image/mask size mismatches: "
        f"{len(size_mismatches)}"
    )

    print(
        f"Empty masks: "
        f"{len(empty_masks)}"
    )

    # --------------------------------------------------------
    # Lesion / nodule size statistics
    # --------------------------------------------------------

    if lesion_ratios:

        ratios = np.array(
            lesion_ratios,
            dtype=np.float64,
        )

        print()
        print("NODULE AREA STATISTICS")
        print("-" * 70)

        print(
            f"Minimum: "
            f"{ratios.min() * 100:.4f}%"
        )

        print(
            f"Median:  "
            f"{np.median(ratios) * 100:.4f}%"
        )

        print(
            f"Mean:    "
            f"{ratios.mean() * 100:.4f}%"
        )

        print(
            f"Maximum: "
            f"{ratios.max() * 100:.4f}%"
        )

        # ----------------------------------------------------
        # Exploratory size categories
        # ----------------------------------------------------

        tiny = np.sum(
            ratios < 0.005
        )

        small = np.sum(
            (ratios >= 0.005)
            & (ratios < 0.02)
        )

        medium = np.sum(
            (ratios >= 0.02)
            & (ratios < 0.10)
        )

        large = np.sum(
            ratios >= 0.10
        )

        print()
        print("EXPLORATORY NODULE SIZE GROUPS")
        print("-" * 70)

        print(
            f"Tiny   (< 0.5%):  "
            f"{tiny}"
        )

        print(
            f"Small  (0.5-2%):  "
            f"{small}"
        )

        print(
            f"Medium (2-10%):   "
            f"{medium}"
        )

        print(
            f"Large  (>=10%):   "
            f"{large}"
        )

    # --------------------------------------------------------
    # Problems
    # --------------------------------------------------------

    if size_mismatches:

        print()
        print(
            "FIRST SIZE MISMATCHES:"
        )

        for mismatch in (
            size_mismatches[:10]
        ):
            print(mismatch)

    if empty_masks:

        print()
        print(
            "FIRST EMPTY MASK IDS:"
        )

        for image_id in empty_masks[:10]:
            print(image_id)


# ============================================================
# OFFICIAL FOLD FILES
# ============================================================

def inspect_fold_json_files():

    print()
    print("=" * 70)
    print("TN3K OFFICIAL FOLD FILES")
    print("=" * 70)

    fold_files = sorted(
        TN3K_ROOT.glob(
            "tn3k-trainval-fold*.json"
        )
    )

    print(
        f"Number of fold JSON files: "
        f"{len(fold_files)}"
    )

    for fold_path in fold_files:

        print()
        print("-" * 70)
        print(fold_path.name)
        print("-" * 70)

        try:

            with open(
                fold_path,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            print(
                f"Python type: "
                f"{type(data).__name__}"
            )

            if isinstance(data, dict):

                print(
                    f"Keys: "
                    f"{list(data.keys())}"
                )

                for key, value in data.items():

                    if isinstance(
                        value,
                        (list, tuple),
                    ):

                        print(
                            f"  {key}: "
                            f"{len(value)} entries"
                        )

                        if len(value) > 0:

                            print(
                                f"    First: "
                                f"{value[0]}"
                            )

                    else:

                        print(
                            f"  {key}: "
                            f"{type(value).__name__}"
                        )

            elif isinstance(data, list):

                print(
                    f"Number of entries: "
                    f"{len(data)}"
                )

                if data:

                    print(
                        f"First entry: "
                        f"{data[0]}"
                    )

        except Exception as error:

            print(
                f"ERROR reading JSON: "
                f"{error}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TN3K THYROID DATASET INSPECTION")
    print("=" * 70)

    print()
    print("TN3K root:")
    print(TN3K_ROOT)

    # --------------------------------------------------------
    # Verify directories
    # --------------------------------------------------------

    required_directories = [
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        TEST_IMAGE_DIR,
        TEST_MASK_DIR,
    ]

    print()
    print("CHECKING REQUIRED FOLDERS")
    print("-" * 70)

    for directory in required_directories:

        if directory.exists():

            print(
                f"OK      {directory}"
            )

        else:

            print(
                f"MISSING {directory}"
            )

            raise FileNotFoundError(
                f"Required TN3K directory "
                f"does not exist:\n"
                f"{directory}"
            )

    # --------------------------------------------------------
    # Read file lists
    # --------------------------------------------------------

    train_images = get_image_files(
        TRAIN_IMAGE_DIR
    )

    train_masks = get_image_files(
        TRAIN_MASK_DIR
    )

    test_images = get_image_files(
        TEST_IMAGE_DIR
    )

    test_masks = get_image_files(
        TEST_MASK_DIR
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FILE COUNTS")
    print("=" * 70)

    print(
        f"Train/validation images: "
        f"{len(train_images)}"
    )

    print(
        f"Train/validation masks:  "
        f"{len(train_masks)}"
    )

    print(
        f"Official test images:    "
        f"{len(test_images)}"
    )

    print(
        f"Official test masks:     "
        f"{len(test_masks)}"
    )

    total_images = (
        len(train_images)
        + len(test_images)
    )

    print(
        f"Total images:             "
        f"{total_images}"
    )

    # --------------------------------------------------------
    # File extensions
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FILE FORMATS")
    print("=" * 70)

    print(
        "Train images:",
        dict(
            get_extension_counts(
                train_images
            )
        ),
    )

    print(
        "Train masks:",
        dict(
            get_extension_counts(
                train_masks
            )
        ),
    )

    print(
        "Test images:",
        dict(
            get_extension_counts(
                test_images
            )
        ),
    )

    print(
        "Test masks:",
        dict(
            get_extension_counts(
                test_masks
            )
        ),
    )

    # --------------------------------------------------------
    # Pair images and masks
    # --------------------------------------------------------

    (
        train_ids,
        train_missing_masks,
        train_orphan_masks,
        train_image_map,
        train_mask_map,
    ) = pair_images_and_masks(
        train_images,
        train_masks,
    )

    (
        test_ids,
        test_missing_masks,
        test_orphan_masks,
        test_image_map,
        test_mask_map,
    ) = pair_images_and_masks(
        test_images,
        test_masks,
    )

    print()
    print("=" * 70)
    print("IMAGE-MASK PAIRING")
    print("=" * 70)

    print()
    print("TRAIN / VALIDATION")

    print(
        f"Matched:       "
        f"{len(train_ids)}"
    )

    print(
        f"Missing masks: "
        f"{len(train_missing_masks)}"
    )

    print(
        f"Orphan masks:  "
        f"{len(train_orphan_masks)}"
    )

    print()
    print("OFFICIAL TEST")

    print(
        f"Matched:       "
        f"{len(test_ids)}"
    )

    print(
        f"Missing masks: "
        f"{len(test_missing_masks)}"
    )

    print(
        f"Orphan masks:  "
        f"{len(test_orphan_masks)}"
    )

    # --------------------------------------------------------
    # Resolutions
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IMAGE RESOLUTIONS")
    print("=" * 70)

    train_resolutions = (
        get_resolution_counts(
            train_images
        )
    )

    test_resolutions = (
        get_resolution_counts(
            test_images
        )
    )

    print()
    print(
        f"Unique train/validation "
        f"resolutions: "
        f"{len(train_resolutions)}"
    )

    for resolution, count in (
        train_resolutions.most_common(20)
    ):

        print(
            f"  {resolution}: "
            f"{count}"
        )

    print()
    print(
        f"Unique official test "
        f"resolutions: "
        f"{len(test_resolutions)}"
    )

    for resolution, count in (
        test_resolutions.most_common(20)
    ):

        print(
            f"  {resolution}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # Detailed split inspection
    # --------------------------------------------------------

    inspect_dataset_split(
        "TRAIN / VALIDATION DATA",
        train_ids,
        train_image_map,
        train_mask_map,
    )

    inspect_dataset_split(
        "OFFICIAL TEST DATA",
        test_ids,
        test_image_map,
        test_mask_map,
    )

    # --------------------------------------------------------
    # Fold JSON files
    # --------------------------------------------------------

    inspect_fold_json_files()

    print()
    print("=" * 70)
    print("TN3K INSPECTION COMPLETE")
    print("=" * 70)

    print(
        "No dataset files were modified."
    )


if __name__ == "__main__":
    main()