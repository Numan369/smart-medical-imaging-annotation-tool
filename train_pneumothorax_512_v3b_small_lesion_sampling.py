"""Train Pneumothorax Model V3B with size-aware positive sampling.

V3B is a controlled continuation of the locked V1 epoch-4 checkpoint. It
retains V1's overall 35% positive / 65% negative training mixture, but divides
the positive portion equally between:

    17.5% tiny/small positive images (ground-truth area < 0.5%)
    17.5% medium/large positive images (ground-truth area >= 0.5%)
    65.0% negative images

Only the distribution within the positive 35% changes. Architecture, loss,
resolution, augmentation, learning rates, trainable layers, threshold,
validation method, maximum epochs and early stopping remain identical to V1.
The test split is never created or accessed.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

import train_pneumothorax_512_negative_aware as training
from pneumothorax_dataset import decode_siim_rle


RANDOM_SEED = 42

EXPECTED_SOURCE_STAGE = "pneumothorax_512_negative_aware_finetune"
EXPECTED_SOURCE_EPOCH = 4
EXPECTED_IMAGE_SIZE = 512
EXPECTED_THRESHOLD = 0.35

EXPECTED_TRAINING_IMAGES = 9637
EXPECTED_TINY_SMALL_POSITIVES = 701
EXPECTED_MEDIUM_LARGE_POSITIVES = 1434
EXPECTED_NEGATIVES = 7502

SOURCE_MASK_SIZE = 1024
SMALL_LESION_MAXIMUM_FRACTION = 0.005

TINY_SMALL_SAMPLE_FRACTION = 0.175
MEDIUM_LARGE_SAMPLE_FRACTION = 0.175
NEGATIVE_SAMPLE_FRACTION = 0.650

DRIVE_PROJECT_DIRECTORY = Path(
    "/content/drive/MyDrive/SmartMedicalImagingColab"
)
SOURCE_CHECKPOINT_PATH = (
    DRIVE_PROJECT_DIRECTORY
    / "checkpoints"
    / "pneumothorax_512_negative_aware_best.pth"
)

# Keep True for the first two-batch Colab smoke test. After it passes, change
# only this value to False and rerun the script for full V3B training.
SMOKE_TEST = True


def almost_equal(first, second):
    return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1e-12)


def validate_inherited_v1_configuration():
    """Refuse to run if any inherited V1 setting has unexpectedly changed."""

    expected_values = {
        "IMAGE_SIZE": 512,
        "BATCH_SIZE": 2,
        "GRADIENT_ACCUMULATION_STEPS": 2,
        "EARLY_STOPPING_PATIENCE": 3,
        "PREDICTION_THRESHOLD": 0.35,
        "TRAINING_POSITIVE_FRACTION": 0.35,
        "DECODER_LEARNING_RATE": 1e-4,
        "ENCODER_4_LEARNING_RATE": 1e-5,
        "ENCODER_3_LEARNING_RATE": 5e-6,
        "WEIGHT_DECAY": 1e-4,
        "POSITIVE_PIXEL_WEIGHT": 4.0,
        "GRADIENT_CLIP_NORM": 1.0,
        "BCE_WEIGHT": 0.45,
        "POSITIVE_TVERSKY_WEIGHT": 0.35,
        "NEGATIVE_BCE_WEIGHT": 0.20,
        "TVERSKY_FALSE_POSITIVE_WEIGHT": 0.50,
        "TVERSKY_FALSE_NEGATIVE_WEIGHT": 0.50,
        "TVERSKY_GAMMA": 0.75,
    }

    for name, expected in expected_values.items():
        actual = getattr(training, name)
        matches = (
            almost_equal(actual, expected)
            if isinstance(expected, float)
            else actual == expected
        )
        if not matches:
            raise ValueError(
                f"Inherited V1 setting {name} is {actual!r}; expected "
                f"{expected!r}. Refusing an uncontrolled experiment."
            )


def validate_source_checkpoint():
    """Refuse any source other than the locked validation-selected V1."""

    if not SOURCE_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Locked V1 checkpoint was not found: {SOURCE_CHECKPOINT_PATH}"
        )

    checkpoint = training.load_torch_checkpoint(
        SOURCE_CHECKPOINT_PATH,
        torch.device("cpu"),
    )
    configuration = checkpoint.get("configuration", {})

    checks = {
        "training stage": (
            checkpoint.get("training_stage"),
            EXPECTED_SOURCE_STAGE,
        ),
        "completed epoch": (
            checkpoint.get("completed_epoch"),
            EXPECTED_SOURCE_EPOCH,
        ),
        "image size": (
            configuration.get("image_size"),
            EXPECTED_IMAGE_SIZE,
        ),
        "prediction threshold": (
            configuration.get("prediction_threshold"),
            EXPECTED_THRESHOLD,
        ),
        "test split used": (
            configuration.get("test_split_used"),
            False,
        ),
    }

    for description, (actual, expected) in checks.items():
        matches = (
            almost_equal(actual, expected)
            if isinstance(expected, float) and actual is not None
            else actual == expected
        )
        if not matches:
            raise ValueError(
                f"Unexpected V1 source {description}: {actual!r}; "
                f"expected {expected!r}."
            )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Locked V1 source has no model_state_dict.")


def unwrap_base_dataset(dataset):
    """Return the unaugmented PneumothoraxDataset used by the wrapper."""

    base_dataset = getattr(dataset, "base_dataset", dataset)

    if getattr(dataset, "split", None) != "train":
        raise ValueError("Size-aware sampling may only use split='train'.")
    if getattr(base_dataset, "split", None) != "train":
        raise ValueError("The wrapped base dataset is not the training split.")
    if len(dataset) != EXPECTED_TRAINING_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_IMAGES:,} training images, "
            f"found {len(dataset):,}."
        )

    return base_dataset


def resized_positive_mask_area(base_dataset, image_id):
    """Calculate the exact 512x512 target area without loading a DICOM."""

    combined_mask = np.zeros(
        (SOURCE_MASK_SIZE, SOURCE_MASK_SIZE),
        dtype=np.uint8,
    )

    encoded_masks = [
        value
        for value in base_dataset.annotations_by_id[image_id]
        if value != "-1"
    ]
    if not encoded_masks:
        raise ValueError(
            f"Positive training image has no positive RLE: {image_id}"
        )

    for encoded_pixels in encoded_masks:
        decoded = decode_siim_rle(
            encoded_pixels,
            SOURCE_MASK_SIZE,
            SOURCE_MASK_SIZE,
        )
        combined_mask = np.maximum(combined_mask, decoded)

    # For the fixed 1024 -> 512 nearest-neighbour resize used by the dataset,
    # output pixel (y, x) reads source pixel (2y, 2x).
    resized_mask = combined_mask[::2, ::2]
    expected_shape = (EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE)
    if resized_mask.shape != expected_shape:
        raise ValueError(
            f"Unexpected resized mask shape: {resized_mask.shape}"
        )

    area = int(resized_mask.sum())
    if area <= 0:
        raise ValueError(
            f"Positive mask disappeared after resizing: {image_id}"
        )
    return area


def classify_training_rows(dataset):
    """Split training rows into tiny/small, medium/large and negative."""

    base_dataset = unwrap_base_dataset(dataset)
    tiny_small_indices = []
    medium_large_indices = []
    negative_indices = []
    image_pixels = EXPECTED_IMAGE_SIZE * EXPECTED_IMAGE_SIZE

    for index, row in enumerate(dataset.rows):
        image_id = str(row["ImageId"]).strip()
        label = int(row["HasPneumothorax"])

        if label == 0:
            negative_indices.append(index)
            continue
        if label != 1:
            raise ValueError(
                f"Training label must be 0 or 1, found {label!r}."
            )

        area = resized_positive_mask_area(base_dataset, image_id)
        area_fraction = area / image_pixels

        if area_fraction < SMALL_LESION_MAXIMUM_FRACTION:
            tiny_small_indices.append(index)
        else:
            medium_large_indices.append(index)

    counts = (
        len(tiny_small_indices),
        len(medium_large_indices),
        len(negative_indices),
    )
    expected_counts = (
        EXPECTED_TINY_SMALL_POSITIVES,
        EXPECTED_MEDIUM_LARGE_POSITIVES,
        EXPECTED_NEGATIVES,
    )
    if counts != expected_counts:
        raise ValueError(
            "Unexpected training size groups. Found tiny-small/"
            f"medium-large/negative={counts}; expected {expected_counts}."
        )

    return tiny_small_indices, medium_large_indices, negative_indices


def create_size_aware_training_sampler(dataset):
    """Sample 17.5% small positives, 17.5% other positives, 65% negatives."""

    (
        tiny_small_indices,
        medium_large_indices,
        negative_indices,
    ) = classify_training_rows(dataset)

    sample_weights = torch.empty(len(dataset), dtype=torch.double)
    sample_weights[tiny_small_indices] = (
        TINY_SMALL_SAMPLE_FRACTION / len(tiny_small_indices)
    )
    sample_weights[medium_large_indices] = (
        MEDIUM_LARGE_SAMPLE_FRACTION / len(medium_large_indices)
    )
    sample_weights[negative_indices] = (
        NEGATIVE_SAMPLE_FRACTION / len(negative_indices)
    )

    if not almost_equal(sample_weights.sum().item(), 1.0):
        raise RuntimeError("V3B sampler weights do not sum to 1.0.")

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    print("V3B size-aware sampler verified")
    print(
        "  Training groups: "
        f"{len(tiny_small_indices):,} tiny/small positive / "
        f"{len(medium_large_indices):,} medium/large positive / "
        f"{len(negative_indices):,} negative"
    )
    print(
        "  Expected sampled mixture: 17.5% tiny/small positive / "
        "17.5% medium/large positive / 65.0% negative"
    )
    print("  Total positive/negative mixture remains 35.0% / 65.0%")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def save_v3b_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_score,
    best_loss,
    epochs_without_improvement,
    training_results,
    validation_results,
):
    """Save V3B state without overwriting V1, V2A or V3A."""

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "training_stage": training.TRAINING_STAGE,
            "completed_epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_selection_score": best_score,
            "best_validation_loss": best_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "training_results": training_results,
            "validation_results": validation_results,
            "configuration": {
                "image_size": training.IMAGE_SIZE,
                "batch_size": training.BATCH_SIZE,
                "gradient_accumulation_steps": (
                    training.GRADIENT_ACCUMULATION_STEPS
                ),
                "prediction_threshold": training.PREDICTION_THRESHOLD,
                "maximum_epochs": training.MAX_EPOCHS,
                "early_stopping_patience": (
                    training.EARLY_STOPPING_PATIENCE
                ),
                "training_positive_fraction": 0.35,
                "tiny_small_positive_sample_fraction": (
                    TINY_SMALL_SAMPLE_FRACTION
                ),
                "medium_large_positive_sample_fraction": (
                    MEDIUM_LARGE_SAMPLE_FRACTION
                ),
                "negative_sample_fraction": NEGATIVE_SAMPLE_FRACTION,
                "small_lesion_maximum_fraction": (
                    SMALL_LESION_MAXIMUM_FRACTION
                ),
                "tiny_small_positive_images": (
                    EXPECTED_TINY_SMALL_POSITIVES
                ),
                "medium_large_positive_images": (
                    EXPECTED_MEDIUM_LARGE_POSITIVES
                ),
                "negative_images": EXPECTED_NEGATIVES,
                "source_checkpoint": str(SOURCE_CHECKPOINT_PATH),
                "source_training_stage": EXPECTED_SOURCE_STAGE,
                "source_checkpoint_epoch": EXPECTED_SOURCE_EPOCH,
                "controlled_change": (
                    "size-aware sampling within positive images only"
                ),
                "augmentation": "paired_conservative_training_only",
                "loss": (
                    "0.45 weighted BCE + 0.35 symmetric positive "
                    "focal Tversky + 0.20 negative-only BCE"
                ),
                "bce_weight": training.BCE_WEIGHT,
                "positive_tversky_weight": (
                    training.POSITIVE_TVERSKY_WEIGHT
                ),
                "negative_bce_weight": training.NEGATIVE_BCE_WEIGHT,
                "positive_pixel_weight": training.POSITIVE_PIXEL_WEIGHT,
                "tversky_false_positive_weight": (
                    training.TVERSKY_FALSE_POSITIVE_WEIGHT
                ),
                "tversky_false_negative_weight": (
                    training.TVERSKY_FALSE_NEGATIVE_WEIGHT
                ),
                "checkpoint_selection": (
                    "harmonic mean of validation positive Dice and "
                    "negative empty-mask accuracy"
                ),
                "validation_split_used": True,
                "test_split_used": False,
            },
        },
        path,
    )


def load_v3b_resume_state(model, optimizer, scheduler, scaler, device):
    """Resume only a checkpoint produced by this exact V3B experiment."""

    checkpoint = training.load_torch_checkpoint(
        training.LAST_CHECKPOINT_PATH,
        device,
    )

    if checkpoint.get("training_stage") != training.TRAINING_STAGE:
        raise ValueError("Resume checkpoint belongs to another training stage.")

    configuration = checkpoint.get("configuration", {})
    required = {
        "controlled_change": (
            "size-aware sampling within positive images only"
        ),
        "source_training_stage": EXPECTED_SOURCE_STAGE,
        "source_checkpoint_epoch": EXPECTED_SOURCE_EPOCH,
        "tiny_small_positive_sample_fraction": (
            TINY_SMALL_SAMPLE_FRACTION
        ),
        "medium_large_positive_sample_fraction": (
            MEDIUM_LARGE_SAMPLE_FRACTION
        ),
        "negative_sample_fraction": NEGATIVE_SAMPLE_FRACTION,
        "small_lesion_maximum_fraction": (
            SMALL_LESION_MAXIMUM_FRACTION
        ),
        "tiny_small_positive_images": EXPECTED_TINY_SMALL_POSITIVES,
        "medium_large_positive_images": EXPECTED_MEDIUM_LARGE_POSITIVES,
        "image_size": EXPECTED_IMAGE_SIZE,
        "prediction_threshold": EXPECTED_THRESHOLD,
        "test_split_used": False,
    }

    for name, expected in required.items():
        actual = configuration.get(name)
        matches = (
            almost_equal(actual, expected)
            if isinstance(expected, float) and actual is not None
            else actual == expected
        )
        if not matches:
            raise ValueError(
                f"Resume setting {name} is {actual!r}; expected "
                f"{expected!r}."
            )

    required_state_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "completed_epoch",
        "best_validation_selection_score",
        "best_validation_loss",
        "epochs_without_improvement",
    }
    missing = required_state_keys - set(checkpoint)
    if missing:
        raise KeyError(
            f"V3B resume checkpoint is missing: {sorted(missing)}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))

    return {
        "start_epoch": checkpoint["completed_epoch"] + 1,
        "best_score": checkpoint["best_validation_selection_score"],
        "best_loss": checkpoint["best_validation_loss"],
        "epochs_without_improvement": checkpoint[
            "epochs_without_improvement"
        ],
    }


def configure_experiment():
    """Apply V3B sampler, paths and safety metadata to the V1 loop."""

    checkpoint_directory = DRIVE_PROJECT_DIRECTORY / "checkpoints"

    training.SMOKE_TEST = SMOKE_TEST
    training.MAX_EPOCHS = 1 if SMOKE_TEST else 8
    training.MAX_TRAINING_BATCHES = 2 if SMOKE_TEST else None
    training.MAX_VALIDATION_BATCHES = 2 if SMOKE_TEST else None
    training.RESUME_IF_AVAILABLE = not SMOKE_TEST

    training.SOURCE_CHECKPOINT_PATH = SOURCE_CHECKPOINT_PATH
    training.LAST_CHECKPOINT_PATH = checkpoint_directory / (
        "pneumothorax_512_v3b_small_lesion_sampling_smoke_last.pth"
        if SMOKE_TEST
        else "pneumothorax_512_v3b_small_lesion_sampling_last.pth"
    )
    training.BEST_CHECKPOINT_PATH = checkpoint_directory / (
        "pneumothorax_512_v3b_small_lesion_sampling_smoke_best.pth"
        if SMOKE_TEST
        else "pneumothorax_512_v3b_small_lesion_sampling_best.pth"
    )
    training.TRAINING_STAGE = (
        "pneumothorax_512_v3b_small_lesion_sampling_smoke_test"
        if SMOKE_TEST
        else "pneumothorax_512_v3b_small_lesion_sampling_finetune"
    )

    training.create_training_sampler = create_size_aware_training_sampler
    training.save_checkpoint = save_v3b_checkpoint
    training.load_resume_state = load_v3b_resume_state


def main():
    validate_inherited_v1_configuration()
    validate_source_checkpoint()
    configure_experiment()

    print("Pneumothorax Model V3B - small-lesion sampling experiment")
    print("----------------------------------------------------------")
    print(f"Run mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL TRAINING'}")
    print(f"Locked V1 source: {SOURCE_CHECKPOINT_PATH}")
    print("Controlled change: positive subgroup sampling only")
    print("  Tiny/small positives: 17.5% of sampled training images")
    print("  Medium/large positives: 17.5%")
    print("  Negatives: unchanged at 65.0%")
    print("  Total positives: unchanged at 35.0%")
    print("V1 loss: unchanged")
    print("Validation split: checkpoint selection only")
    print("Test split: not created or accessed\n")

    training.main()


if __name__ == "__main__":
    main()
