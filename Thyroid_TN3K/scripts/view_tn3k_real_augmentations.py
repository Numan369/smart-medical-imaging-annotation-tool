import matplotlib.pyplot as plt
import numpy as np
import torch

from tn3k_dataset import TN3KDataset
from tn3k_augmentation import TN3KTrainAugmentation


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_ID = "trainval_2011"

NUMBER_OF_AUGMENTATIONS = 6


# ============================================================
# FIND SAMPLE INDEX
# ============================================================

def find_sample_index(
    dataset,
    sample_id,
):

    for index, row in enumerate(
        dataset.rows
    ):

        if (
            row["SampleId"]
            == sample_id
        ):

            return index

    raise ValueError(
        f"Could not find "
        f"{sample_id}"
    )


# ============================================================
# SHOW OVERLAY
# ============================================================

def show_overlay(
    axis,
    sample,
    title,
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

    axis.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    overlay = np.ma.masked_where(
        mask < 0.5,
        mask,
    )

    axis.imshow(
        overlay,
        alpha=0.40,
    )

    axis.set_title(title)

    axis.axis("off")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("REAL TN3K AUGMENTATION VISUAL CHECK")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset without augmentation
    # --------------------------------------------------------

    base_dataset = TN3KDataset(
        split="train",
        image_size=512,
        augmentation=None,
    )

    # --------------------------------------------------------
    # Dataset WITH augmentation
    # --------------------------------------------------------

    augmenter = (
        TN3KTrainAugmentation()
    )

    augmented_dataset = TN3KDataset(
        split="train",
        image_size=512,
        augmentation=augmenter,
    )

    # --------------------------------------------------------
    # Same image index in both datasets
    # --------------------------------------------------------

    index = find_sample_index(
        base_dataset,
        SAMPLE_ID,
    )

    original = base_dataset[index]

    print()
    print(
        f"Sample: "
        f"{original['sample_id']}"
    )

    print(
        f"Nodule size group: "
        f"{original['nodule_size_group']}"
    )

    print(
        f"Original area: "
        f"{original['nodule_area_fraction'] * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
    )

    axes = axes.flatten()

    # Original
    show_overlay(
        axes[0],
        original,
        "Original\nNo Augmentation",
    )

    # --------------------------------------------------------
    # Repeated random augmentations
    # --------------------------------------------------------

    for i in range(
        NUMBER_OF_AUGMENTATIONS
    ):

        sample = augmented_dataset[
            index
        ]

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        image = sample["image"]
        mask = sample["mask"]

        assert tuple(
            image.shape
        ) == (
            1,
            512,
            512,
        )

        assert tuple(
            mask.shape
        ) == (
            1,
            512,
            512,
        )

        assert torch.isfinite(
            image
        ).all()

        assert torch.isfinite(
            mask
        ).all()

        unique_mask = set(
            torch.unique(
                mask
            ).tolist()
        )

        assert unique_mask.issubset(
            {
                0.0,
                1.0,
            }
        )

        assert float(
            mask.sum()
        ) > 0

        info = sample[
            "augmentation"
        ]

        print()
        print(
            f"Augmentation "
            f"{i + 1}"
        )

        print(
            f"  Affine: "
            f"{info['affine_applied']}"
        )

        print(
            f"  Angle: "
            f"{info['angle']:.2f}"
        )

        print(
            f"  Translation: "
            f"("
            f"{info['translate_x']}, "
            f"{info['translate_y']}"
            f")"
        )

        print(
            f"  Scale: "
            f"{info['scale']:.4f}"
        )

        print(
            f"  Brightness: "
            f"{info['brightness_factor']:.4f}"
        )

        print(
            f"  Contrast: "
            f"{info['contrast_factor']:.4f}"
        )

        print(
            f"  Mask pixels: "
            f"{int(mask.sum().item())}"
        )

        show_overlay(
            axes[i + 1],
            sample,
            (
                f"Augmentation {i + 1}\n"
                f"{info['angle']:.1f}°"
            ),
        )

    # --------------------------------------------------------
    # Hide unused panel
    # --------------------------------------------------------

    for index_unused in range(
        NUMBER_OF_AUGMENTATIONS + 1,
        len(axes),
    ):

        axes[
            index_unused
        ].axis("off")

    figure.suptitle(
        "TN3K V1 — Real Training Augmentation Check",
        fontsize=15,
    )

    plt.tight_layout()

    plt.show()

    print()
    print("=" * 70)
    print(
        "REAL AUGMENTATION CHECK PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()