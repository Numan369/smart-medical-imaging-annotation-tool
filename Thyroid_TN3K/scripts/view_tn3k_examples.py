from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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

IMAGE_DIR = TN3K_ROOT / "trainval-image"
MASK_DIR = TN3K_ROOT / "trainval-mask"


# ============================================================
# EXAMPLE IDS
# ============================================================

# We deliberately start with the same examples that appeared
# in our inspection output.
EXAMPLE_IDS = [
    "0000",
    "0001",
    "0002",
    "0003",
    "0004",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TN3K IMAGE + MASK VISUAL INSPECTION")
    print("=" * 70)

    for image_id in EXAMPLE_IDS:

        image_path = IMAGE_DIR / f"{image_id}.jpg"
        mask_path = MASK_DIR / f"{image_id}.jpg"

        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue

        if not mask_path.exists():
            print(f"Missing mask: {mask_path}")
            continue

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        image = np.array(
            Image.open(image_path).convert("L")
        )

        mask = np.array(
            Image.open(mask_path).convert("L")
        )

        # ----------------------------------------------------
        # Pixel statistics
        # ----------------------------------------------------

        unique_values = np.unique(mask)

        print()
        print("-" * 70)
        print(f"Image ID: {image_id}")
        print("-" * 70)

        print(f"Image shape: {image.shape}")
        print(f"Mask shape:  {mask.shape}")

        print()
        print("MASK PIXEL STATISTICS")

        print(f"Minimum mask value: {mask.min()}")
        print(f"Maximum mask value: {mask.max()}")

        print(
            f"Number of unique mask values: "
            f"{len(unique_values)}"
        )

        if len(unique_values) <= 30:

            print(
                "Unique values:",
                unique_values.tolist(),
            )

        else:

            print(
                "First 15 values:",
                unique_values[:15].tolist(),
            )

            print(
                "Last 15 values:",
                unique_values[-15:].tolist(),
            )

        # ----------------------------------------------------
        # Compare thresholds
        # ----------------------------------------------------

        thresholds = [
            0,
            10,
            50,
            127,
            200,
        ]

        print()
        print("MASK AREA AT DIFFERENT THRESHOLDS")

        for threshold in thresholds:

            binary = mask > threshold

            ratio = (
                binary.sum()
                / binary.size
            )

            print(
                f"mask > {threshold:3d}: "
                f"{ratio * 100:.4f}%"
            )

        # ----------------------------------------------------
        # Use 127 only for this VISUALIZATION.
        #
        # We are NOT yet locking 127 as the final training
        # threshold.
        # ----------------------------------------------------

        binary_mask = mask > 127

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = np.zeros(
            (
                image.shape[0],
                image.shape[1],
                4,
            ),
            dtype=np.float32,
        )

        overlay[..., 0] = 1.0

        overlay[..., 3] = (
            binary_mask.astype(np.float32)
            * 0.40
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        figure, axes = plt.subplots(
            1,
            4,
            figsize=(16, 5),
        )

        # Original ultrasound
        axes[0].imshow(
            image,
            cmap="gray",
        )

        axes[0].set_title(
            f"Ultrasound\n{image_id}"
        )

        axes[0].axis("off")

        # Raw JPEG mask
        axes[1].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        axes[1].set_title(
            "Raw JPEG Mask"
        )

        axes[1].axis("off")

        # Binary version
        axes[2].imshow(
            binary_mask,
            cmap="gray",
        )

        axes[2].set_title(
            "Binary Mask\n(threshold > 127)"
        )

        axes[2].axis("off")

        # Overlay
        axes[3].imshow(
            image,
            cmap="gray",
        )

        axes[3].imshow(
            overlay,
        )

        axes[3].set_title(
            "Nodule Mask Overlay"
        )

        axes[3].axis("off")

        figure.suptitle(
            f"TN3K Example {image_id}",
            fontsize=14,
        )

        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()