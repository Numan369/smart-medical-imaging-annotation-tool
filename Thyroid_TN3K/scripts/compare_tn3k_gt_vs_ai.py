from pathlib import Path
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from infer_tn3k_v1 import (
    DEFAULT_CHECKPOINT,
    get_device,
    preprocess_image,
    load_model,
    predict_model_space,
    restore_mask_to_original_size,
)


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

TEST_IMAGE_DIR = (
    TN3K_ROOT
    / "test-image"
)

TRAINVAL_IMAGE_DIR = (
    TN3K_ROOT
    / "trainval-image"
)

DEFAULT_OUTPUT_DIR = (
    THYROID_ROOT
    / "outputs"
    / "tn3k_gt_vs_ai"
)

DEFAULT_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

MASK_THRESHOLD = 128
EPSILON = 1e-6


# ============================================================
# RANDOM IMAGE SELECTION
# ============================================================

def choose_random_image(
    split="test",
):
    """
    Select a random TN3K image.

    split:
        "test"     -> official test-image directory
        "trainval" -> trainval-image directory
    """

    if split == "test":

        image_dir = TEST_IMAGE_DIR

    elif split == "trainval":

        image_dir = TRAINVAL_IMAGE_DIR

    else:

        raise ValueError(
            f"Unsupported split: {split}"
        )


    if not image_dir.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n"
            f"{image_dir}"
        )


    image_files = sorted(
        list(
            image_dir.glob("*.jpg")
        )
        +
        list(
            image_dir.glob("*.jpeg")
        )
        +
        list(
            image_dir.glob("*.JPG")
        )
        +
        list(
            image_dir.glob("*.JPEG")
        )
    )


    if not image_files:

        raise RuntimeError(
            f"No JPG images found in:\n"
            f"{image_dir}"
        )


    selected_image = random.choice(
        image_files
    )


    return selected_image


# ============================================================
# FIND CORRESPONDING GROUND-TRUTH MASK
# ============================================================

def find_ground_truth_mask(
    image_path,
):

    image_path = Path(
        image_path
    )


    image_parent_name = (
        image_path.parent.name
    )


    if image_parent_name == "test-image":

        mask_dir = (
            image_path.parent.parent
            / "test-mask"
        )


    elif image_parent_name == "trainval-image":

        mask_dir = (
            image_path.parent.parent
            / "trainval-mask"
        )


    else:

        raise ValueError(
            "Expected a TN3K image from:\n"
            "  test-image\n"
            "or\n"
            "  trainval-image\n\n"
            f"Received:\n"
            f"{image_path}"
        )


    mask_path = (
        mask_dir
        / image_path.name
    )


    if not mask_path.exists():

        raise FileNotFoundError(
            "Corresponding ground-truth "
            "mask not found:\n"
            f"{mask_path}"
        )


    return mask_path


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth_mask(
    mask_path,
    expected_width,
    expected_height,
):

    mask_path = Path(
        mask_path
    )


    raw_mask = Image.open(
        mask_path
    ).convert(
        "L"
    )


    if (
        raw_mask.width
        != expected_width

        or

        raw_mask.height
        != expected_height
    ):

        raise RuntimeError(
            "Image/mask size mismatch.\n"
            f"Expected: "
            f"{expected_width}"
            f"x"
            f"{expected_height}\n"
            f"Mask: "
            f"{raw_mask.width}"
            f"x"
            f"{raw_mask.height}"
        )


    raw_mask_array = np.array(
        raw_mask,
        dtype=np.uint8,
    )


    binary_mask = (
        raw_mask_array
        >= MASK_THRESHOLD
    ).astype(
        np.uint8
    )


    return binary_mask


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    ground_truth,
    prediction,
):

    ground_truth = (
        ground_truth > 0
    )

    prediction = (
        prediction > 0
    )


    true_positive = np.logical_and(
        ground_truth,
        prediction,
    ).sum()


    false_positive = np.logical_and(
        ~ground_truth,
        prediction,
    ).sum()


    false_negative = np.logical_and(
        ground_truth,
        ~prediction,
    ).sum()


    dice = (
        2.0 * true_positive
        + EPSILON
    ) / (
        2.0 * true_positive
        + false_positive
        + false_negative
        + EPSILON
    )


    iou = (
        true_positive
        + EPSILON
    ) / (
        true_positive
        + false_positive
        + false_negative
        + EPSILON
    )


    precision = (
        true_positive
        + EPSILON
    ) / (
        true_positive
        + false_positive
        + EPSILON
    )


    recall = (
        true_positive
        + EPSILON
    ) / (
        true_positive
        + false_negative
        + EPSILON
    )


    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


# ============================================================
# CREATE OVERLAP
# ============================================================

def create_overlap_image(
    original_image,
    ground_truth,
    prediction,
):

    grayscale = (
        original_image.astype(
            np.float32
        )
        / 255.0
    )


    comparison = np.stack(
        [
            grayscale,
            grayscale,
            grayscale,
        ],
        axis=-1,
    )


    gt = (
        ground_truth > 0
    )


    pred = (
        prediction > 0
    )


    correct_overlap = np.logical_and(
        gt,
        pred,
    )


    missed_gt = np.logical_and(
        gt,
        ~pred,
    )


    extra_prediction = np.logical_and(
        pred,
        ~gt,
    )


    alpha = 0.65


    # Yellow = correct overlap
    comparison[
        correct_overlap
    ] = (
        (1.0 - alpha)
        * comparison[
            correct_overlap
        ]
        +
        alpha
        * np.array(
            [1.0, 1.0, 0.0]
        )
    )


    # Green = GT missed by model
    comparison[
        missed_gt
    ] = (
        (1.0 - alpha)
        * comparison[
            missed_gt
        ]
        +
        alpha
        * np.array(
            [0.0, 1.0, 0.0]
        )
    )


    # Red = extra AI prediction
    comparison[
        extra_prediction
    ] = (
        (1.0 - alpha)
        * comparison[
            extra_prediction
        ]
        +
        alpha
        * np.array(
            [1.0, 0.0, 0.0]
        )
    )


    return np.clip(
        comparison,
        0.0,
        1.0,
    )


# ============================================================
# VISUALIZATION
# ============================================================

def save_comparison_figure(
    original_image,
    ground_truth,
    prediction,
    overlap_image,
    metrics,
    image_name,
    output_path,
):

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5),
    )


    # 1 ------------------------------------------------------

    axes[0].imshow(
        original_image,
        cmap="gray",
    )

    axes[0].set_title(
        "Original Ultrasound"
    )

    axes[0].axis(
        "off"
    )


    # 2 ------------------------------------------------------

    axes[1].imshow(
        ground_truth,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "Ground Truth Mask"
    )

    axes[1].axis(
        "off"
    )


    # 3 ------------------------------------------------------

    axes[2].imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        "AI Predicted Mask"
    )

    axes[2].axis(
        "off"
    )


    # 4 ------------------------------------------------------

    axes[3].imshow(
        overlap_image
    )

    axes[3].set_title(
        "GT vs AI Overlap"
    )

    axes[3].axis(
        "off"
    )


    figure.suptitle(

        f"TN3K Ground Truth vs AI — "
        f"{image_name}\n"

        f"Dice={metrics['dice']:.4f} | "
        f"IoU={metrics['iou']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f}",

        fontsize=14,
    )


    figure.text(

        0.5,
        0.02,

        "YELLOW = Correct overlap   |   "
        "GREEN = Ground truth missed by AI   |   "
        "RED = Extra AI prediction",

        ha="center",

        fontsize=11,
    )


    plt.tight_layout(
        rect=[
            0,
            0.06,
            1,
            0.92,
        ]
    )


    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )


    plt.show()


# ============================================================
# MAIN COMPARISON
# ============================================================

def compare_ground_truth_vs_ai(
    image_path,
    checkpoint_path=DEFAULT_CHECKPOINT,
    output_dir=DEFAULT_OUTPUT_DIR,
):

    image_path = Path(
        image_path
    )


    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print("=" * 70)
    print(
        "TN3K RANDOM GROUND TRUTH VS AI COMPARISON"
    )
    print("=" * 70)


    device = get_device()


    print()
    print(
        "Device:",
        device
    )


    (
        tensor,
        metadata,
        original_image,
    ) = preprocess_image(
        image_path
    )


    print()
    print(
        "Selected image:"
    )

    print(
        image_path
    )


    mask_path = find_ground_truth_mask(
        image_path
    )


    print()
    print(
        "Ground-truth mask:"
    )

    print(
        mask_path
    )


    ground_truth = load_ground_truth_mask(

        mask_path,

        metadata[
            "original_width"
        ],

        metadata[
            "original_height"
        ],
    )


    model, _ = load_model(

        checkpoint_path,

        device,
    )


    (
        _,
        model_prediction,
    ) = predict_model_space(

        model,

        tensor,

        device,
    )


    restored_prediction = (
        restore_mask_to_original_size(

            model_prediction,

            metadata,
        )
    )


    metrics = calculate_metrics(

        ground_truth,

        restored_prediction,
    )


    overlap_image = create_overlap_image(

        original_image,

        ground_truth,

        restored_prediction,
    )


    output_path = (

        output_dir

        / (
            f"{image_path.stem}"
            f"_gt_vs_ai.png"
        )
    )


    save_comparison_figure(

        original_image,

        ground_truth,

        restored_prediction,

        overlap_image,

        metrics,

        image_path.name,

        output_path,
    )


    print()
    print("=" * 70)
    print(
        "COMPARISON COMPLETE"
    )
    print("=" * 70)


    print()
    print(
        "Image:",
        image_path.name
    )


    print(
        f"Dice:      "
        f"{metrics['dice']:.6f}"
    )


    print(
        f"IoU:       "
        f"{metrics['iou']:.6f}"
    )


    print(
        f"Precision: "
        f"{metrics['precision']:.6f}"
    )


    print(
        f"Recall:    "
        f"{metrics['recall']:.6f}"
    )


    print()
    print(
        "Saved:"
    )

    print(
        output_path
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Randomly compare TN3K ground truth "
            "against the locked final AI model."
        )
    )


    # --------------------------------------------------------
    # Image is OPTIONAL now.
    #
    # If not supplied -> random image.
    # --------------------------------------------------------

    parser.add_argument(

        "image",

        nargs="?",

        default=None,

        help=(
            "Optional specific TN3K image. "
            "If omitted, a random image is selected."
        ),
    )


    parser.add_argument(

        "--split",

        choices=[
            "test",
            "trainval",
        ],

        default="test",

        help=(
            "Dataset to randomly sample from "
            "when no image is supplied."
        ),
    )


    parser.add_argument(

        "--checkpoint",

        default=str(
            DEFAULT_CHECKPOINT
        ),
    )


    parser.add_argument(

        "--output-dir",

        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )


    args = parser.parse_args()


    # ========================================================
    # RANDOM OR MANUAL
    # ========================================================

    if args.image is None:

        image_path = choose_random_image(
            split=args.split
        )


        print()
        print(
            "Random image selected:"
        )

        print(
            image_path
        )


    else:

        image_path = Path(
            args.image
        )


        print()
        print(
            "Manually selected image:"
        )

        print(
            image_path
        )


    compare_ground_truth_vs_ai(

        image_path=image_path,

        checkpoint_path=args.checkpoint,

        output_dir=args.output_dir,
    )


if __name__ == "__main__":

    main()