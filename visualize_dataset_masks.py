from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pneumothorax_dataset import PneumothoraxDataset


OUTPUT_PATH = Path("dataset_mask_check.png")
NUMBER_OF_SAMPLES = 3
CANDIDATES_TO_CHECK = 100


def main():
    dataset = PneumothoraxDataset(
        split="train",
        image_size=256,
    )

    positive_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if row["HasPneumothorax"] == 1
    ]

    candidates = []

    print("Searching for clearly visible masks...")

    for index in positive_indices[:CANDIDATES_TO_CHECK]:
        sample = dataset[index]
        mask_pixels = int(sample["mask"].sum().item())

        candidates.append(
            (mask_pixels, sample)
        )

    # Select the three largest masks among the checked samples.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    selected_samples = [
        sample
        for _, sample in candidates[:NUMBER_OF_SAMPLES]
    ]

    figure, axes = plt.subplots(
        nrows=NUMBER_OF_SAMPLES,
        ncols=3,
        figsize=(11, 9),
        layout="constrained",
    )

    figure.suptitle(
        "SIIM Pneumothorax Mask Verification",
        fontsize=16,
    )

    for row_number, sample in enumerate(selected_samples):
        image = sample["image"].squeeze(0).numpy()
        mask = sample["mask"].squeeze(0).numpy()
        image_id = sample["image_id"]
        mask_pixels = int(mask.sum())

        axes[row_number, 0].imshow(
            image,
            cmap="gray",
        )
        axes[row_number, 0].set_title(
            f"X-ray\nID ending: {image_id[-12:]}"
        )

        axes[row_number, 1].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row_number, 1].set_title(
            f"Binary mask\n{mask_pixels} pixels"
        )

        axes[row_number, 2].imshow(
            image,
            cmap="gray",
        )

        axes[row_number, 2].imshow(
            np.ma.masked_where(mask == 0, mask),
            cmap="autumn",
            alpha=0.8,
            vmin=0,
            vmax=1,
        )

        # Add a bright yellow boundary to make the mask obvious.
        axes[row_number, 2].contour(
            mask,
            levels=[0.5],
            colors=["yellow"],
            linewidths=1.5,
        )

        axes[row_number, 2].set_title(
            "Red mask with yellow boundary"
        )

        for column_number in range(3):
            axes[row_number, column_number].axis("off")

        print(
            f"Selected: {image_id} "
            f"({mask_pixels} marked pixels)"
        )

    figure.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    print("\nMask visualization complete")
    print("---------------------------")
    print(f"Saved image: {OUTPUT_PATH.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()