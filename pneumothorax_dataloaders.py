import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from pneumothorax_dataset import (
    IMAGE_SIZE,
    PneumothoraxDataset,
)


BATCH_SIZE = 4
NUM_WORKERS = 0
RANDOM_SEED = 42


def get_dataset_labels(dataset):
    """Return the binary image label stored for every dataset row."""

    labels = torch.tensor(
        [
            int(row["HasPneumothorax"])
            for row in dataset.rows
        ],
        dtype=torch.long,
    )

    unexpected_labels = set(labels.tolist()) - {0, 1}

    if unexpected_labels:
        raise ValueError(
            "Training labels must be 0 or 1. "
            f"Found: {sorted(unexpected_labels)}"
        )

    return labels


def create_balanced_training_sampler(train_dataset):
    """Sample positive and negative training images equally often."""

    labels = get_dataset_labels(train_dataset)

    positive_count = int(labels.sum().item())
    negative_count = len(labels) - positive_count

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Balanced sampling requires both positive and "
            "negative training images."
        )

    # Each class receives the same total probability. Individual
    # samples in the smaller positive class are therefore selected
    # more frequently than individual negative samples.
    class_weights = torch.tensor(
        [
            1.0 / negative_count,
            1.0 / positive_count,
        ],
        dtype=torch.double,
    )
    sample_weights = class_weights[labels]

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(RANDOM_SEED)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=sampler_generator,
    )


def create_dataloaders(
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
):
    """Create balanced training and ordinary evaluation DataLoaders."""

    train_dataset = PneumothoraxDataset(
        split="train",
        image_size=image_size,
    )

    validation_dataset = PneumothoraxDataset(
        split="validation",
        image_size=image_size,
    )

    test_dataset = PneumothoraxDataset(
        split="test",
        image_size=image_size,
    )

    training_sampler = create_balanced_training_sampler(
        train_dataset
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=training_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    # Validation and test sets must keep their natural class
    # distributions so that their metrics remain honest.
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }


def check_batch(split_name, data_loader):
    """Load and validate the first batch from one DataLoader."""

    batch = next(iter(data_loader))

    images = batch["image"]
    masks = batch["mask"]
    labels = batch["label"]
    image_ids = batch["image_id"]

    expected_shape = (
        images.shape[0],
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if tuple(images.shape) != expected_shape:
        raise ValueError(
            f"Unexpected image shape: {tuple(images.shape)}"
        )

    if tuple(masks.shape) != expected_shape:
        raise ValueError(
            f"Unexpected mask shape: {tuple(masks.shape)}"
        )

    if images.min() < 0 or images.max() > 1:
        raise ValueError(
            "Image values are outside the 0-1 range."
        )

    mask_values = set(torch.unique(masks).tolist())

    if not mask_values.issubset({0.0, 1.0}):
        raise ValueError(
            f"Mask contains unexpected values: {mask_values}"
        )

    print(f"\n{split_name.capitalize()} batch")
    print(f"  Number of images: {images.shape[0]}")
    print(f"  Image batch shape: {tuple(images.shape)}")
    print(f"  Mask batch shape: {tuple(masks.shape)}")
    print(f"  Label batch shape: {tuple(labels.shape)}")
    print(
        "  Positive images in batch: "
        f"{int(labels.sum().item())}"
    )
    print(
        "  Annotated mask pixels: "
        f"{int(masks.sum().item())}"
    )
    print(f"  First image ID: {image_ids[0]}")


def print_dataset_class_counts(split_name, dataset):
    """Print the original positive and negative image counts."""

    labels = get_dataset_labels(dataset)
    positive_count = int(labels.sum().item())
    negative_count = len(labels) - positive_count

    print(
        f"{split_name.capitalize()} original labels: "
        f"{positive_count} positive, "
        f"{negative_count} negative"
    )


def print_sampled_training_counts(train_loader):
    """Show the labels selected for one balanced training epoch."""

    sampled_indices = list(iter(train_loader.sampler))
    sampled_labels = [
        int(
            train_loader.dataset.rows[int(index)][
                "HasPneumothorax"
            ]
        )
        for index in sampled_indices
    ]

    positive_count = sum(sampled_labels)
    negative_count = len(sampled_labels) - positive_count

    print(
        "Balanced training sample for one epoch: "
        f"{positive_count} positive, "
        f"{negative_count} negative"
    )


if __name__ == "__main__":
    data_loaders = create_dataloaders()

    print("DataLoader check")
    print("----------------")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Worker processes: {NUM_WORKERS}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")

    for split_name, data_loader in data_loaders.items():
        print(
            f"{split_name.capitalize()} dataset size: "
            f"{len(data_loader.dataset)}"
        )
        print(
            f"{split_name.capitalize()} batches: "
            f"{len(data_loader)}"
        )
        print_dataset_class_counts(
            split_name,
            data_loader.dataset,
        )

    print_sampled_training_counts(data_loaders["train"])

    for split_name, data_loader in data_loaders.items():
        check_batch(
            split_name,
            data_loader,
        )

    print("\nAll DataLoaders passed their checks.")