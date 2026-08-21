from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PATHS
# ============================================================

THYROID_ROOT = Path(__file__).resolve().parent.parent

INDEX_CSV = (
    THYROID_ROOT
    / "prepared_data"
    / "tn3k_dataset_index.csv"
)

OUTPUT_DIR = (
    THYROID_ROOT
    / "outputs"
    / "resize_comparison"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_SIZE = 512
MASK_THRESHOLD = 128


# ============================================================
# LOAD INDEX
# ============================================================

def load_training_examples():

    rows = []

    with open(
        INDEX_CSV,
        "r",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["Split"] == "train":
                rows.append(row)

    return rows


def choose_examples(rows):

    """
    Select one TRAINING example from each
    nodule-size category.
    """

    selected = {}

    for group in [
        "tiny",
        "small",
        "medium",
        "large",
    ]:

        candidates = [
            row
            for row in rows
            if row["NoduleSizeGroup"] == group
        ]

        if not candidates:
            continue

        # Choose approximately the middle example
        # instead of always the first one.
        selected[group] = (
            candidates[len(candidates) // 2]
        )

    return selected


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image_and_mask(row):

    image_path = (
        THYROID_ROOT
        / row["ImagePath"]
    )

    mask_path = (
        THYROID_ROOT
        / row["MaskPath"]
    )

    image = Image.open(
        image_path
    ).convert("L")

    raw_mask = Image.open(
        mask_path
    ).convert("L")

    mask_array = np.array(
        raw_mask
    )

    # Correct TN3K JPEG-mask conversion
    binary_mask = (
        mask_array >= MASK_THRESHOLD
    ).astype(np.uint8) * 255

    mask = Image.fromarray(
        binary_mask
    )

    return image, mask


# ============================================================
# STRATEGY 1:
# DIRECT RESIZE
# ============================================================

def direct_resize(image, mask):

    resized_image = image.resize(
        (TARGET_SIZE, TARGET_SIZE),
        Image.Resampling.BILINEAR,
    )

    resized_mask = mask.resize(
        (TARGET_SIZE, TARGET_SIZE),
        Image.Resampling.NEAREST,
    )

    return resized_image, resized_mask


# ============================================================
# STRATEGY 2:
# ASPECT-RATIO-PRESERVING RESIZE + PAD
# ============================================================

def letterbox_resize(image, mask):

    original_width, original_height = (
        image.size
    )

    scale = min(
        TARGET_SIZE / original_width,
        TARGET_SIZE / original_height,
    )

    new_width = int(
        round(original_width * scale)
    )

    new_height = int(
        round(original_height * scale)
    )

    # --------------------------------------------------------
    # Resize while preserving aspect ratio
    # --------------------------------------------------------

    resized_image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR,
    )

    resized_mask = mask.resize(
        (new_width, new_height),
        Image.Resampling.NEAREST,
    )

    # --------------------------------------------------------
    # Create square canvases
    # --------------------------------------------------------

    image_canvas = Image.new(
        "L",
        (TARGET_SIZE, TARGET_SIZE),
        color=0,
    )

    mask_canvas = Image.new(
        "L",
        (TARGET_SIZE, TARGET_SIZE),
        color=0,
    )

    # --------------------------------------------------------
    # Center the resized data
    # --------------------------------------------------------

    left = (
        TARGET_SIZE - new_width
    ) // 2

    top = (
        TARGET_SIZE - new_height
    ) // 2

    image_canvas.paste(
        resized_image,
        (left, top),
    )

    mask_canvas.paste(
        resized_mask,
        (left, top),
    )

    return (
        image_canvas,
        mask_canvas,
        new_width,
        new_height,
        left,
        top,
    )


# ============================================================
# OVERLAY
# ============================================================

def make_overlay(image, mask):

    image_array = np.array(
        image
    )

    mask_array = np.array(
        mask
    ) > 127

    overlay = np.zeros(
        (
            image_array.shape[0],
            image_array.shape[1],
            4,
        ),
        dtype=np.float32,
    )

    # red overlay
    overlay[..., 0] = 1.0

    overlay[..., 3] = (
        mask_array.astype(
            np.float32
        )
        * 0.40
    )

    return image_array, overlay


# ============================================================
# VISUALIZATION
# ============================================================

def show_example(
    size_group,
    row,
):

    image, mask = (
        load_image_and_mask(row)
    )

    direct_image, direct_mask = (
        direct_resize(
            image,
            mask,
        )
    )

    (
        letter_image,
        letter_mask,
        new_width,
        new_height,
        left,
        top,
    ) = letterbox_resize(
        image,
        mask,
    )

    original_image_array, original_overlay = (
        make_overlay(
            image,
            mask,
        )
    )

    direct_array, direct_overlay = (
        make_overlay(
            direct_image,
            direct_mask,
        )
    )

    letter_array, letter_overlay = (
        make_overlay(
            letter_image,
            letter_mask,
        )
    )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"{size_group.upper()} NODULE"
    )
    print("=" * 70)

    print(
        f"Sample ID: "
        f"{row['SampleId']}"
    )

    print(
        f"Original size: "
        f"{image.size}"
    )

    print(
        f"Original nodule area: "
        f"{row['NoduleAreaPercent']}%"
    )

    print()
    print(
        "Direct resize:"
    )

    print(
        f"  {image.size} "
        f"-> "
        f"({TARGET_SIZE}, "
        f"{TARGET_SIZE})"
    )

    print()
    print(
        "Aspect-preserving resize:"
    )

    print(
        f"  {image.size} "
        f"-> "
        f"({new_width}, "
        f"{new_height})"
    )

    print(
        f"  Padding offset: "
        f"left={left}, "
        f"top={top}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    # Original
    axes[0].imshow(
        original_image_array,
        cmap="gray",
    )

    axes[0].imshow(
        original_overlay
    )

    axes[0].set_title(
        f"Original\n"
        f"{image.size[0]} × "
        f"{image.size[1]}"
    )

    axes[0].axis("off")

    # Direct square resize
    axes[1].imshow(
        direct_array,
        cmap="gray",
    )

    axes[1].imshow(
        direct_overlay
    )

    axes[1].set_title(
        "Direct Resize\n"
        "512 × 512"
    )

    axes[1].axis("off")

    # Letterbox
    axes[2].imshow(
        letter_array,
        cmap="gray",
    )

    axes[2].imshow(
        letter_overlay
    )

    axes[2].set_title(
        "Preserve Aspect Ratio\n"
        "+ Pad to 512 × 512"
    )

    axes[2].axis("off")

    figure.suptitle(
        f"TN3K Resize Comparison — "
        f"{size_group.upper()} — "
        f"{row['SampleId']}",
        fontsize=14,
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save copy
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"{size_group}_"
            f"{row['SampleId']}.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print()
    print(
        f"Saved comparison:"
    )

    print(output_path)

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TN3K RESIZE STRATEGY COMPARISON")
    print("=" * 70)

    print()
    print(
        f"Target model size: "
        f"{TARGET_SIZE} x {TARGET_SIZE}"
    )

    rows = load_training_examples()

    print(
        f"Training examples loaded: "
        f"{len(rows)}"
    )

    selected = choose_examples(
        rows
    )

    print()
    print(
        "Selected training examples:"
    )

    for group, row in (
        selected.items()
    ):

        print(
            f"  {group:<6} "
            f"{row['SampleId']} "
            f"area="
            f"{row['NoduleAreaPercent']}%"
        )

    for group, row in (
        selected.items()
    ):

        show_example(
            group,
            row,
        )

    print()
    print("=" * 70)
    print("RESIZE COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()