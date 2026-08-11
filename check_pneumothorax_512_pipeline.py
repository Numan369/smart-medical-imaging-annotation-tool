import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pneumothorax_dataloaders import create_balanced_training_sampler
from pneumothorax_dataset import PneumothoraxDataset


OLD_IMAGE_SIZE = 256
NEW_IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_SMALLEST_CASES = 5
RANDOM_SEED = 42

FAILURE_DETAILS_PATH = (
    Path("validation_analysis") / "positive_case_details.csv"
)


def load_smallest_validation_ids():
    """Read the smallest positive cases found by failure analysis."""

    if not FAILURE_DETAILS_PATH.is_file():
        raise FileNotFoundError(
            "Failure-analysis CSV was not found at: "
            f"{FAILURE_DETAILS_PATH.resolve()}\n"
            "Run analyze_pneumothorax_validation_failures.py first."
        )

    with FAILURE_DETAILS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as details_file:
        rows = list(csv.DictReader(details_file))

    required_columns = {
        "image_id",
        "target_area_percent",
    }

    if not rows:
        raise ValueError(
            "The failure-analysis CSV contains no positive cases."
        )

    missing_columns = required_columns - set(rows[0])

    if missing_columns:
        raise ValueError(
            "The failure-analysis CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    rows.sort(
        key=lambda row: float(row["target_area_percent"])
    )

    return [
        row["image_id"]
        for row in rows[:NUM_SMALLEST_CASES]
    ]


def index_by_image_id(dataset):
    """Map each dataset image ID to its integer dataset index."""

    return {
        row["ImageId"]: index
        for index, row in enumerate(dataset.rows)
    }


def inspect_small_masks():
    """Compare tiny expert masks at 256 and 512 pixels."""

    image_ids = load_smallest_validation_ids()

    validation_256 = PneumothoraxDataset(
        split="validation",
        image_size=OLD_IMAGE_SIZE,
    )
    validation_512 = PneumothoraxDataset(
        split="validation",
        image_size=NEW_IMAGE_SIZE,
    )

    indices_256 = index_by_image_id(validation_256)
    indices_512 = index_by_image_id(validation_512)

    print("\nSmallest positive validation masks")
    print("Image ID | 256 pixels | 512 pixels | Scale")

    for image_id in image_ids:
        if image_id not in indices_256 or image_id not in indices_512:
            raise ValueError(
                "A failure-analysis image is missing from the "
                f"validation split: {image_id}"
            )

        mask_256 = validation_256[
            indices_256[image_id]
        ]["mask"]
        mask_512 = validation_512[
            indices_512[image_id]
        ]["mask"]

        pixels_256 = int(mask_256.sum().item())
        pixels_512 = int(mask_512.sum().item())

        if pixels_256 <= 0 or pixels_512 <= 0:
            raise ValueError(
                "A known positive mask became empty after resizing: "
                f"{image_id}"
            )

        scale = pixels_512 / pixels_256

        print(
            f"{image_id} | {pixels_256} | "
            f"{pixels_512} | {scale:.2f}x"
        )


def validate_batch(split_name, batch):
    """Check shapes, ranges, labels, and binary masks."""

    images = batch["image"]
    masks = batch["mask"]
    labels = batch["label"]

    expected_shape = (
        images.shape[0],
        1,
        NEW_IMAGE_SIZE,
        NEW_IMAGE_SIZE,
    )

    if tuple(images.shape) != expected_shape:
        raise ValueError(
            f"Unexpected {split_name} image shape: "
            f"{tuple(images.shape)}"
        )

    if tuple(masks.shape) != expected_shape:
        raise ValueError(
            f"Unexpected {split_name} mask shape: "
            f"{tuple(masks.shape)}"
        )

    if images.min().item() < 0.0 or images.max().item() > 1.0:
        raise ValueError(
            f"{split_name.capitalize()} images are outside 0-1."
        )

    mask_values = set(torch.unique(masks).tolist())

    if not mask_values.issubset({0.0, 1.0}):
        raise ValueError(
            f"{split_name.capitalize()} masks are not binary: "
            f"{sorted(mask_values)}"
        )

    tensor_bytes = (
        images.numel() * images.element_size()
        + masks.numel() * masks.element_size()
    )

    print(f"\n{split_name.capitalize()} batch check")
    print(f"Images: {tuple(images.shape)}")
    print(f"Masks: {tuple(masks.shape)}")
    print(
        "Positive images: "
        f"{int(labels.sum().item())} / {len(labels)}"
    )
    print(f"Annotated pixels: {int(masks.sum().item()):,}")
    print(
        "Input image + mask memory: "
        f"{tensor_bytes / (1024 ** 2):.2f} MiB"
    )


def main():
    torch.manual_seed(RANDOM_SEED)

    print("512 x 512 pneumothorax data-pipeline check")
    print("------------------------------------------")
    print(f"Image size: {NEW_IMAGE_SIZE} x {NEW_IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Dataset splits used: train and validation only")
    print("Test split: not created or accessed")

    inspect_small_masks()

    training_dataset = PneumothoraxDataset(
        split="train",
        image_size=NEW_IMAGE_SIZE,
    )
    validation_dataset = PneumothoraxDataset(
        split="validation",
        image_size=NEW_IMAGE_SIZE,
    )

    training_sampler = create_balanced_training_sampler(
        training_dataset
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        sampler=training_sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    print("\nDataset summary")
    print(f"Training images: {len(training_dataset):,}")
    print(f"Validation images: {len(validation_dataset):,}")
    print(f"Training batches: {len(training_loader):,}")
    print(f"Validation batches: {len(validation_loader):,}")

    validate_batch(
        "training",
        next(iter(training_loader)),
    )
    validate_batch(
        "validation",
        next(iter(validation_loader)),
    )

    print("\n## 512 x 512 pipeline check passed")
    print(
        "The dataset can now supply binary masks and normalized "
        "X-rays at the upgraded resolution."
    )
    print("No training was performed and no checkpoint was changed.")


if __name__ == "__main__":
    main()
