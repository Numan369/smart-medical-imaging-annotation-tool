from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from tn3k_dataset import TN3KDataset


# ============================================================
# PATHS
# ============================================================

THYROID_ROOT = (
    Path(__file__).resolve().parent.parent
)

DATASET_ROOT = (
    THYROID_ROOT
    / "dataset"
    / "Thyroid Dataset"
    / "tn3k"
)

RAW_IMAGE_DIR = (
    DATASET_ROOT
    / "trainval-image"
)

RAW_MASK_DIR = (
    DATASET_ROOT
    / "trainval-mask"
)


OUTPUT_DIR = (
    THYROID_ROOT
    / "checkpoints"
    / "tn3k_v1_earlystop"
    / "validation_diagnostics"
    / "raw_pair_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CASES TO AUDIT
# ============================================================

WORST_SAMPLE_IDS = [

    "trainval_1920",

    "trainval_0570",

    "trainval_0705",

    "trainval_2050",

    "trainval_0885",

    "trainval_1090",
]


IMAGE_SIZE = 512

MASK_THRESHOLD = 128


# ============================================================
# FIND VALIDATION SAMPLE
# ============================================================

def find_validation_index(
    dataset,
    sample_id,
):

    for index in range(
        len(dataset)
    ):

        sample = dataset[
            index
        ]


        if (
            sample["sample_id"]
            == sample_id
        ):

            return index


    raise ValueError(
        f"Could not find "
        f"{sample_id} "
        f"in validation dataset."
    )


# ============================================================
# FIND RAW FILE BY ORIGINAL ID
# ============================================================

def find_raw_file(
    directory,
    original_id,
):
    """
    TN3K files are JPG images.

    This function is deliberately tolerant of:
        1920.jpg
        01920.jpg
        etc.

    It first tries direct filename matching,
    then compares numeric stems.
    """

    original_id_string = str(
        original_id
    )


    # --------------------------------------------------------
    # First try exact stem.
    # --------------------------------------------------------

    direct_candidates = [

        directory
        / f"{original_id_string}.jpg",

        directory
        / f"{original_id_string}.jpeg",

        directory
        / f"{original_id_string}.JPG",

        directory
        / f"{original_id_string}.JPEG",
    ]


    for path in direct_candidates:

        if path.exists():

            return path


    # --------------------------------------------------------
    # Try numeric comparison.
    # --------------------------------------------------------

    try:

        target_number = int(
            original_id_string
        )

    except ValueError:

        target_number = None


    for path in directory.iterdir():

        if not path.is_file():

            continue


        if path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
        }:

            continue


        if path.stem == original_id_string:

            return path


        if target_number is not None:

            try:

                candidate_number = int(
                    path.stem
                )

            except ValueError:

                continue


            if (
                candidate_number
                == target_number
            ):

                return path


    raise FileNotFoundError(
        f"Could not find raw file for "
        f"OriginalId={original_id}\n"
        f"Directory={directory}"
    )


# ============================================================
# RAW MASK PREPARATION
# ============================================================

def load_raw_mask(
    path,
):

    raw_mask = np.array(
        Image.open(
            path
        ).convert(
            "L"
        ),
        dtype=np.uint8,
    )


    binary_mask = (
        raw_mask
        >= MASK_THRESHOLD
    ).astype(
        np.uint8
    )


    return (
        raw_mask,
        binary_mask,
    )


# ============================================================
# RAW OVERLAY
# ============================================================

def create_raw_overlay(
    raw_image,
    binary_mask,
):

    figure, axis = plt.subplots(
        1,
        1,
        figsize=(6, 6),
    )


    axis.imshow(
        raw_image,
        cmap="gray",
    )


    if binary_mask.max() > 0:

        axis.contour(
            binary_mask,
            levels=[0.5],
            linewidths=2,
            colors="lime",
        )


    axis.set_title(
        "RAW Image + RAW GT"
    )


    axis.axis(
        "off"
    )


    return (
        figure,
        axis,
    )


# ============================================================
# COMPLETE AUDIT FIGURE
# ============================================================

def create_audit_figure(
    sample_id,
    original_id,
    raw_image_path,
    raw_mask_path,
    processed_sample,
):

    # ========================================================
    # RAW IMAGE
    # ========================================================

    raw_image = np.array(
        Image.open(
            raw_image_path
        ).convert(
            "L"
        ),
        dtype=np.uint8,
    )


    # ========================================================
    # RAW MASK
    # ========================================================

    (
        raw_mask,
        raw_binary_mask,
    ) = load_raw_mask(
        raw_mask_path
    )


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if (
        raw_image.shape
        != raw_mask.shape
    ):

        raise RuntimeError(
            f"RAW SIZE MISMATCH for {sample_id}\n"
            f"Image: {raw_image.shape}\n"
            f"Mask:  {raw_mask.shape}"
        )


    raw_positive_pixels = int(
        raw_binary_mask.sum()
    )


    if raw_positive_pixels == 0:

        raise RuntimeError(
            f"RAW mask became empty after "
            f">= {MASK_THRESHOLD} threshold "
            f"for {sample_id}"
        )


    # ========================================================
    # PROCESSED DATASET OUTPUT
    # ========================================================

    processed_image = (
        processed_sample[
            "image"
        ][0]
        .cpu()
        .numpy()
    )


    processed_mask = (
        processed_sample[
            "mask"
        ][0]
        .cpu()
        .numpy()
    )


    processed_positive_pixels = int(
        processed_mask.sum()
    )


    # ========================================================
    # FIGURE
    # ========================================================

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(20, 10),
    )


    # --------------------------------------------------------
    # 1. Raw image
    # --------------------------------------------------------

    axes[0, 0].imshow(
        raw_image,
        cmap="gray",
    )


    axes[0, 0].set_title(
        f"RAW Ultrasound\n"
        f"{raw_image.shape[1]}×"
        f"{raw_image.shape[0]}"
    )


    axes[0, 0].axis(
        "off"
    )


    # --------------------------------------------------------
    # 2. Raw JPEG mask values
    # --------------------------------------------------------

    raw_mask_display = (
        axes[0, 1].imshow(
            raw_mask,
            cmap="gray",
            vmin=0,
            vmax=255,
        )
    )


    axes[0, 1].set_title(
        "RAW JPEG Mask"
    )


    axes[0, 1].axis(
        "off"
    )


    figure.colorbar(
        raw_mask_display,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04,
    )


    # --------------------------------------------------------
    # 3. Thresholded raw mask
    # --------------------------------------------------------

    axes[0, 2].imshow(
        raw_binary_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


    axes[0, 2].set_title(
        f"RAW Mask >= {MASK_THRESHOLD}\n"
        f"Pixels={raw_positive_pixels:,}"
    )


    axes[0, 2].axis(
        "off"
    )


    # --------------------------------------------------------
    # 4. Raw image + GT contour
    # --------------------------------------------------------

    axes[0, 3].imshow(
        raw_image,
        cmap="gray",
    )


    axes[0, 3].contour(
        raw_binary_mask,
        levels=[0.5],
        linewidths=2,
        colors="lime",
    )


    axes[0, 3].set_title(
        "RAW Image + GT"
    )


    axes[0, 3].axis(
        "off"
    )


    # --------------------------------------------------------
    # 5. Processed image
    # --------------------------------------------------------

    axes[1, 0].imshow(
        processed_image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


    axes[1, 0].set_title(
        "Processed Ultrasound\n512×512"
    )


    axes[1, 0].axis(
        "off"
    )


    # --------------------------------------------------------
    # 6. Processed mask
    # --------------------------------------------------------

    axes[1, 1].imshow(
        processed_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


    axes[1, 1].set_title(
        "Processed GT Mask\n"
        f"Pixels={processed_positive_pixels:,}"
    )


    axes[1, 1].axis(
        "off"
    )


    # --------------------------------------------------------
    # 7. Processed overlay
    # --------------------------------------------------------

    axes[1, 2].imshow(
        processed_image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


    if processed_mask.max() > 0:

        axes[1, 2].contour(
            processed_mask,
            levels=[0.5],
            linewidths=2,
            colors="lime",
        )


    axes[1, 2].set_title(
        "Processed Image + GT"
    )


    axes[1, 2].axis(
        "off"
    )


    # --------------------------------------------------------
    # 8. Information
    # --------------------------------------------------------

    axes[1, 3].axis(
        "off"
    )


    information = (

        f"Sample ID:\n"
        f"{sample_id}\n\n"

        f"Original ID:\n"
        f"{original_id}\n\n"

        f"Raw image:\n"
        f"{raw_image_path.name}\n\n"

        f"Raw mask:\n"
        f"{raw_mask_path.name}\n\n"

        f"Raw size:\n"
        f"{raw_image.shape[1]}"
        f" × "
        f"{raw_image.shape[0]}\n\n"

        f"Mask threshold:\n"
        f">= {MASK_THRESHOLD}\n\n"

        f"Raw positive pixels:\n"
        f"{raw_positive_pixels:,}\n\n"

        f"Processed positive pixels:\n"
        f"{processed_positive_pixels:,}\n\n"

        f"Size group:\n"
        f"{processed_sample['nodule_size_group']}"
    )


    axes[1, 3].text(

        0.02,
        0.98,

        information,

        va="top",
        ha="left",

        fontsize=11,

        family="monospace",
    )


    figure.suptitle(

        f"TN3K RAW/PROCESSED PAIR AUDIT — "
        f"{sample_id}",

        fontsize=16,
    )


    plt.tight_layout()


    output_path = (

        OUTPUT_DIR
        /
        (
            f"audit_"
            f"{sample_id}.png"
        )
    )


    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


    return {

        "sample_id":
            sample_id,

        "original_id":
            original_id,

        "raw_width":
            raw_image.shape[1],

        "raw_height":
            raw_image.shape[0],

        "raw_positive_pixels":
            raw_positive_pixels,

        "processed_positive_pixels":
            processed_positive_pixels,

        "size_group":
            processed_sample[
                "nodule_size_group"
            ],

        "output_path":
            output_path,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "TN3K WORST-CASE RAW IMAGE/MASK PAIR AUDIT"
    )
    print("=" * 70)


    print()
    print(
        "This script does NOT train the model."
    )

    print(
        "Official TN3K test set is NOT used."
    )


    # ========================================================
    # PATH CHECKS
    # ========================================================

    print()
    print(
        "Raw image directory:"
    )

    print(
        RAW_IMAGE_DIR
    )


    print()
    print(
        "Raw mask directory:"
    )

    print(
        RAW_MASK_DIR
    )


    if not RAW_IMAGE_DIR.exists():

        raise FileNotFoundError(
            RAW_IMAGE_DIR
        )


    if not RAW_MASK_DIR.exists():

        raise FileNotFoundError(
            RAW_MASK_DIR
        )


    # ========================================================
    # LOAD FOLD-0 VALIDATION DATASET
    # ========================================================

    validation_dataset = TN3KDataset(

        split="validation",

        image_size=IMAGE_SIZE,

        augmentation=None,
    )


    if (
        len(validation_dataset)
        != 576
    ):

        raise RuntimeError(
            f"Expected 576 validation images, "
            f"found "
            f"{len(validation_dataset)}."
        )


    print()
    print(
        "Validation samples:",
        len(validation_dataset)
    )


    # ========================================================
    # AUDIT CASES
    # ========================================================

    results = []


    for sample_id in WORST_SAMPLE_IDS:

        print()
        print("-" * 70)

        print(
            "Auditing:",
            sample_id
        )


        # ----------------------------------------------------
        # Find sample in actual validation dataset.
        # ----------------------------------------------------

        dataset_index = find_validation_index(

            validation_dataset,

            sample_id,
        )


        sample = validation_dataset[
            dataset_index
        ]


        original_id = (
            sample[
                "original_id"
            ]
        )


        print(
            "Validation dataset index:",
            dataset_index
        )


        print(
            "Original ID:",
            original_id
        )


        # ----------------------------------------------------
        # Find raw image and mask.
        # ----------------------------------------------------

        raw_image_path = find_raw_file(

            RAW_IMAGE_DIR,

            original_id,
        )


        raw_mask_path = find_raw_file(

            RAW_MASK_DIR,

            original_id,
        )


        print(
            "Raw image:",
            raw_image_path.name
        )


        print(
            "Raw mask:",
            raw_mask_path.name
        )


        # ----------------------------------------------------
        # Audit figure
        # ----------------------------------------------------

        result = create_audit_figure(

            sample_id=sample_id,

            original_id=original_id,

            raw_image_path=raw_image_path,

            raw_mask_path=raw_mask_path,

            processed_sample=sample,
        )


        results.append(
            result
        )


        print(
            "Raw size:",
            f"{result['raw_width']}"
            f"×"
            f"{result['raw_height']}"
        )


        print(
            "Raw GT pixels:",
            result[
                "raw_positive_pixels"
            ]
        )


        print(
            "Processed GT pixels:",
            result[
                "processed_positive_pixels"
            ]
        )


        print(
            "Saved:",
            result[
                "output_path"
            ].name
        )


    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 70)
    print(
        "RAW PAIR AUDIT COMPLETE"
    )
    print("=" * 70)


    print()
    print(
        "Cases audited:",
        len(results)
    )


    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )


    print()
    print(
        "Official TN3K test set was NOT used."
    )


if __name__ == "__main__":

    main()