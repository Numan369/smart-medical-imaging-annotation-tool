import torch

from torch.utils.data import DataLoader

from tn3k_dataset import TN3KDataset
from tn3k_augmentation import TN3KTrainAugmentation


# ============================================================
# TN3K V1 SETTINGS
# ============================================================

IMAGE_SIZE = 512

# Local Windows test configuration.
#
# We already know batch size 2 is safe from our
# pneumothorax 512x512 work.
BATCH_SIZE = 2

# Windows + local debugging:
# workers=0 is the safest configuration.
NUM_WORKERS = 0


# ============================================================
# CREATE DATASETS
# ============================================================

def create_tn3k_datasets():

    # --------------------------------------------------------
    # TRAINING AUGMENTATION
    # --------------------------------------------------------

    train_augmentation = (
        TN3KTrainAugmentation()
    )

    # --------------------------------------------------------
    # TRAIN
    #
    # Augmentation ON
    # --------------------------------------------------------

    train_dataset = TN3KDataset(
        split="train",
        image_size=IMAGE_SIZE,
        augmentation=train_augmentation,
    )

    # --------------------------------------------------------
    # VALIDATION
    #
    # Augmentation OFF
    # --------------------------------------------------------

    validation_dataset = TN3KDataset(
        split="validation",
        image_size=IMAGE_SIZE,
        augmentation=None,
    )

    return (
        train_dataset,
        validation_dataset,
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_tn3k_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
):

    (
        train_dataset,
        validation_dataset,
    ) = create_tn3k_datasets()

    # --------------------------------------------------------
    # TRAIN LOADER
    #
    # shuffle=True:
    # Each epoch sees training samples in a new order.
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=False,

        drop_last=False,
    )

    # --------------------------------------------------------
    # VALIDATION LOADER
    #
    # shuffle=False:
    # Validation must stay deterministic and stable.
    # --------------------------------------------------------

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=False,

        drop_last=False,
    )

    return (
        train_loader,
        validation_loader,
    )


# ============================================================
# CHECK ONE BATCH
# ============================================================

def inspect_batch(
    batch,
    name,
):

    print()
    print("=" * 70)
    print(f"{name.upper()} BATCH")
    print("=" * 70)

    images = batch["image"]
    masks = batch["mask"]

    print(
        f"Image batch shape: "
        f"{tuple(images.shape)}"
    )

    print(
        f"Mask batch shape:  "
        f"{tuple(masks.shape)}"
    )

    print(
        f"Image dtype: "
        f"{images.dtype}"
    )

    print(
        f"Mask dtype:  "
        f"{masks.dtype}"
    )

    print(
        f"Image minimum: "
        f"{images.min().item():.4f}"
    )

    print(
        f"Image maximum: "
        f"{images.max().item():.4f}"
    )

    print(
        f"Mask unique values: "
        f"{torch.unique(masks).tolist()}"
    )

    print()
    print(
        f"Sample IDs: "
        f"{list(batch['sample_id'])}"
    )

    print(
        f"Nodule groups: "
        f"{list(batch['nodule_size_group'])}"
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    assert images.ndim == 4
    assert masks.ndim == 4

    assert images.shape[1:] == (
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert masks.shape[1:] == (
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert images.shape[0] <= BATCH_SIZE
    assert masks.shape[0] <= BATCH_SIZE

    assert torch.isfinite(
        images
    ).all()

    assert torch.isfinite(
        masks
    ).all()

    assert images.min() >= 0.0
    assert images.max() <= 1.0

    unique_mask_values = set(
        torch.unique(
            masks
        ).tolist()
    )

    assert unique_mask_values.issubset(
        {
            0.0,
            1.0,
        }
    )

    # Every TN3K image should contain
    # a nodule mask.
    for sample_mask in masks:

        assert (
            sample_mask.sum()
            > 0
        )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("TN3K DATALOADER TEST")
    print("=" * 70)

    (
        train_loader,
        validation_loader,
    ) = create_tn3k_dataloaders()

    print()
    print(
        f"Training dataset size: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation dataset size: "
        f"{len(validation_loader.dataset)}"
    )

    print()
    print(
        f"Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    # --------------------------------------------------------
    # First training batch
    # --------------------------------------------------------

    train_batch = next(
        iter(train_loader)
    )

    inspect_batch(
        train_batch,
        "train",
    )

    # --------------------------------------------------------
    # First validation batch
    # --------------------------------------------------------

    validation_batch = next(
        iter(validation_loader)
    )

    inspect_batch(
        validation_batch,
        "validation",
    )

    # --------------------------------------------------------
    # Verify validation has NO augmentation
    # --------------------------------------------------------

    validation_aug = (
        validation_batch[
            "augmentation"
        ]
    )

    print()
    print("VALIDATION AUGMENTATION CHECK")
    print("-" * 70)

    print(
        "Affine applied:",
        validation_aug[
            "affine_applied"
        ].tolist(),
    )

    print(
        "Angles:",
        validation_aug[
            "angle"
        ].tolist(),
    )

    print(
        "Brightness applied:",
        validation_aug[
            "brightness_applied"
        ].tolist(),
    )

    print(
        "Contrast applied:",
        validation_aug[
            "contrast_applied"
        ].tolist(),
    )

    assert not validation_aug[
        "affine_applied"
    ].any()

    assert not validation_aug[
        "brightness_applied"
    ].any()

    assert not validation_aug[
        "contrast_applied"
    ].any()

    print()
    print(
        "Validation augmentation: OFF"
    )

    print()
    print("=" * 70)
    print(
        "TN3K DATALOADER TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()