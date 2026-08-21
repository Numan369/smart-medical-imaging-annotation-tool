import matplotlib.pyplot as plt
import numpy as np

from tn3k_dataset import TN3KDataset


TARGET_IDS = {
    "train": "trainval_0048",
    "validation": "trainval_2825",
}


def find_sample(dataset, sample_id):

    for index in range(len(dataset)):

        sample = dataset[index]

        if sample["sample_id"] == sample_id:
            return sample

    raise ValueError(
        f"Could not find sample: {sample_id}"
    )


def show_sample(sample):

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

    geometry = sample["geometry"]

    content_fraction = (
        geometry["resized_width"]
        * geometry["resized_height"]
        / (512 * 512)
    )

    padding_fraction = (
        1.0 - content_fraction
    )

    print()
    print("=" * 70)

    print(
        f"Sample: {sample['sample_id']}"
    )

    print(
        f"Original size: "
        f"{geometry['original_width']} × "
        f"{geometry['original_height']}"
    )

    print(
        f"Resized content: "
        f"{geometry['resized_width']} × "
        f"{geometry['resized_height']}"
    )

    print(
        f"Padding: "
        f"{padding_fraction * 100:.2f}%"
    )

    print(
        f"Nodule group: "
        f"{sample['nodule_size_group']}"
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5),
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title(
        "512×512 Ultrasound"
    )

    axes[0].axis("off")

    axes[1].imshow(
        mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "Binary Nodule Mask"
    )

    axes[1].axis("off")

    axes[2].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    overlay = np.ma.masked_where(
        mask < 0.5,
        mask,
    )

    axes[2].imshow(
        overlay,
        alpha=0.4,
    )

    axes[2].set_title(
        "Mask Overlay"
    )

    axes[2].axis("off")

    figure.suptitle(
        f"{sample['sample_id']} — "
        f"Padding {padding_fraction * 100:.2f}%"
    )

    plt.tight_layout()
    plt.show()


def main():

    train_dataset = TN3KDataset(
        split="train",
        image_size=512,
    )

    validation_dataset = TN3KDataset(
        split="validation",
        image_size=512,
    )

    train_sample = find_sample(
        train_dataset,
        TARGET_IDS["train"],
    )

    validation_sample = find_sample(
        validation_dataset,
        TARGET_IDS["validation"],
    )

    show_sample(train_sample)
    show_sample(validation_sample)


if __name__ == "__main__":
    main()