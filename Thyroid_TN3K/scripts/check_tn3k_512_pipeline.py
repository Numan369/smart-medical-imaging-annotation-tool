from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from tn3k_dataset import TN3KDataset


# ============================================================
# PATHS / SETTINGS
# ============================================================

THYROID_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    THYROID_ROOT
    / "outputs"
    / "pipeline_check"
)

IMAGE_SIZE = 512


# ============================================================
# CHECK ONE SAMPLE
# ============================================================

def validate_sample(sample):

    image = sample["image"]
    mask = sample["mask"]

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    if tuple(image.shape) != (
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise ValueError(
            f"Bad image shape for "
            f"{sample['sample_id']}: "
            f"{tuple(image.shape)}"
        )

    if tuple(mask.shape) != (
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise ValueError(
            f"Bad mask shape for "
            f"{sample['sample_id']}: "
            f"{tuple(mask.shape)}"
        )

    # --------------------------------------------------------
    # Finite values
    # --------------------------------------------------------

    if not torch.isfinite(image).all():
        raise ValueError(
            f"Non-finite image values: "
            f"{sample['sample_id']}"
        )

    if not torch.isfinite(mask).all():
        raise ValueError(
            f"Non-finite mask values: "
            f"{sample['sample_id']}"
        )

    # --------------------------------------------------------
    # Image range
    # --------------------------------------------------------

    image_min = float(image.min())
    image_max = float(image.max())

    if image_min < 0.0 or image_max > 1.0:

        raise ValueError(
            f"Image outside [0,1]: "
            f"{sample['sample_id']} "
            f"min={image_min} "
            f"max={image_max}"
        )

    # --------------------------------------------------------
    # Binary mask
    # --------------------------------------------------------

    unique_mask = set(
        torch.unique(mask).tolist()
    )

    if not unique_mask.issubset(
        {0.0, 1.0}
    ):

        raise ValueError(
            f"Non-binary mask: "
            f"{sample['sample_id']} "
            f"{sorted(unique_mask)}"
        )

    if float(mask.sum()) <= 0:

        raise ValueError(
            f"Empty transformed mask: "
            f"{sample['sample_id']}"
        )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    geometry = sample["geometry"]

    resized_width = int(
        geometry["resized_width"]
    )

    resized_height = int(
        geometry["resized_height"]
    )

    if (
        resized_width > IMAGE_SIZE
        or resized_height > IMAGE_SIZE
    ):

        raise ValueError(
            f"Bad resized dimensions: "
            f"{sample['sample_id']}"
        )

    # One dimension should reach exactly 512.
    if (
        resized_width != IMAGE_SIZE
        and resized_height != IMAGE_SIZE
    ):

        raise ValueError(
            f"Neither resized dimension reaches "
            f"{IMAGE_SIZE}: "
            f"{sample['sample_id']}"
        )

    content_fraction = (
        resized_width
        * resized_height
        / (IMAGE_SIZE * IMAGE_SIZE)
    )

    padding_fraction = (
        1.0 - content_fraction
    )

    return padding_fraction


# ============================================================
# FULL SPLIT CHECK
# ============================================================

def scan_dataset(
    dataset,
    split_name,
):

    print()
    print("=" * 70)
    print(
        f"SCANNING {split_name.upper()}"
    )
    print("=" * 70)

    max_padding = -1.0
    max_padding_id = None

    padding_values = []

    for index in range(
        len(dataset)
    ):

        sample = dataset[index]

        padding_fraction = (
            validate_sample(sample)
        )

        padding_values.append(
            padding_fraction
        )

        if (
            padding_fraction
            > max_padding
        ):

            max_padding = (
                padding_fraction
            )

            max_padding_id = (
                sample["sample_id"]
            )

        if (
            (index + 1) % 500 == 0
            or index + 1 == len(dataset)
        ):

            print(
                f"Checked "
                f"{index + 1}/"
                f"{len(dataset)}"
            )

    padding_values = np.array(
        padding_values,
        dtype=np.float64,
    )

    print()
    print("RESULTS")
    print("-" * 70)

    print(
        f"Samples checked: "
        f"{len(dataset)}"
    )

    print(
        f"Mean padding fraction: "
        f"{padding_values.mean() * 100:.2f}%"
    )

    print(
        f"Median padding fraction: "
        f"{np.median(padding_values) * 100:.2f}%"
    )

    print(
        f"Maximum padding fraction: "
        f"{max_padding * 100:.2f}%"
    )

    print(
        f"Maximum-padding sample: "
        f"{max_padding_id}"
    )

    return padding_values


# ============================================================
# CHOOSE TRAINING EXAMPLES
# ============================================================

def choose_group_examples(
    dataset,
):

    selected = {}

    for group in [
        "tiny",
        "small",
        "medium",
        "large",
    ]:

        indices = [
            index
            for index, row
            in enumerate(dataset.rows)
            if (
                row["NoduleSizeGroup"]
                == group
            )
        ]

        if not indices:
            continue

        # Choose middle candidate
        selected[group] = (
            indices[len(indices) // 2]
        )

    return selected


# ============================================================
# VISUALIZE ACTUAL DATASET TENSORS
# ============================================================

def visualize_sample(
    sample,
    group,
):

    image = (
        sample["image"][0]
        .cpu()
        .numpy()
    )

    mask = (
        sample["mask"][0]
        .cpu()
        .numpy()
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4.5),
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title(
        "512 × 512 Ultrasound"
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    axes[1].imshow(
        mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "512 × 512 Binary Mask"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    axes[2].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    masked_overlay = np.ma.masked_where(
        mask < 0.5,
        mask,
    )

    axes[2].imshow(
        masked_overlay,
        alpha=0.40,
    )

    axes[2].set_title(
        "Dataset Tensor Overlay"
    )

    axes[2].axis("off")

    geometry = sample["geometry"]

    figure.suptitle(
        f"{group.upper()} — "
        f"{sample['sample_id']}\n"
        f"Original "
        f"{geometry['original_width']}×"
        f"{geometry['original_height']} "
        f"→ resized "
        f"{geometry['resized_width']}×"
        f"{geometry['resized_height']} "
        f"→ 512×512",
        fontsize=12,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"{group}_"
            f"{sample['sample_id']}.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"Saved: {output_path}"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TN3K 512 PIPELINE CHECK")
    print("=" * 70)

    train_dataset = TN3KDataset(
        split="train",
        image_size=512,
    )

    validation_dataset = TN3KDataset(
        split="validation",
        image_size=512,
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    if len(train_dataset) != 2303:

        raise ValueError(
            "Unexpected training count."
        )

    if len(validation_dataset) != 576:

        raise ValueError(
            "Unexpected validation count."
        )

    # --------------------------------------------------------
    # Scan all development samples
    # --------------------------------------------------------

    scan_dataset(
        train_dataset,
        "train",
    )

    scan_dataset(
        validation_dataset,
        "validation",
    )

    # --------------------------------------------------------
    # Visual examples
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "VISUALIZING TRAINING EXAMPLES"
    )
    print("=" * 70)

    selected = (
        choose_group_examples(
            train_dataset
        )
    )

    for group, index in (
        selected.items()
    ):

        sample = train_dataset[
            index
        ]

        print()
        print(
            f"{group.upper()}: "
            f"{sample['sample_id']}"
        )

        visualize_sample(
            sample,
            group,
        )

    print()
    print("=" * 70)
    print(
        "TN3K 512 PIPELINE CHECK PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()