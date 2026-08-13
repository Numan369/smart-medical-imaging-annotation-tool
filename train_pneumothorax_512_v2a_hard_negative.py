"""Train Pneumothorax Model V2A with hard-negative sampling.

This is a controlled continuation of the locked epoch-4 V1 model. It reuses
the existing negative-aware training implementation and changes only how the
negative part of the training sampler is divided:

    35.0% positive images
    32.5% mined hard-negative images
    32.5% other negative images

The total 35% positive / 65% negative mixture, architecture, loss,
augmentation, trainable layers, learning rates, validation procedure, image
size, and prediction threshold remain unchanged. The test split is never
created or accessed.
"""

import math
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler

import train_pneumothorax_512_negative_aware as training


RANDOM_SEED = 42
EXPECTED_TRAINING_IMAGES = 9637
EXPECTED_TRAINING_POSITIVES = 2135
EXPECTED_TRAINING_NEGATIVES = 7502
EXPECTED_HARD_NEGATIVES = 2141
EXPECTED_OTHER_NEGATIVES = 5361

POSITIVE_SAMPLE_FRACTION = 0.35
HARD_NEGATIVE_SAMPLE_FRACTION = 0.325
OTHER_NEGATIVE_SAMPLE_FRACTION = 0.325

EXPECTED_SOURCE_STAGE = "pneumothorax_512_negative_aware_finetune"
EXPECTED_SOURCE_EPOCH = 4
EXPECTED_IMAGE_SIZE = 512
EXPECTED_THRESHOLD = 0.35

DRIVE_PROJECT_DIRECTORY = Path(
    "/content/drive/MyDrive/SmartMedicalImagingColab"
)
SOURCE_CHECKPOINT_PATH = (
    DRIVE_PROJECT_DIRECTORY
    / "checkpoints"
    / "pneumothorax_512_negative_aware_best.pth"
)
HARD_NEGATIVE_IDS_PATH = (
    DRIVE_PROJECT_DIRECTORY
    / "hard_negative_mining_v2"
    / "hard_negative_ids.txt"
)

# Set this to True only for a quick smoke test. Full V2A resume runs should
# use False so the script can continue from the saved epoch-5 checkpoint.
SMOKE_TEST = False


def almost_equal(first, second):
    return math.isclose(first, second, rel_tol=0.0, abs_tol=1e-12)


def validate_unchanged_training_configuration():
    """Refuse to run if the inherited V1 training recipe has drifted."""

    expected_values = {
        "IMAGE_SIZE": 512,
        "BATCH_SIZE": 2,
        "GRADIENT_ACCUMULATION_STEPS": 2,
        "PREDICTION_THRESHOLD": 0.35,
        "TRAINING_POSITIVE_FRACTION": 0.35,
        "DECODER_LEARNING_RATE": 1e-4,
        "ENCODER_4_LEARNING_RATE": 1e-5,
        "ENCODER_3_LEARNING_RATE": 5e-6,
        "BCE_WEIGHT": 0.45,
        "POSITIVE_TVERSKY_WEIGHT": 0.35,
        "NEGATIVE_BCE_WEIGHT": 0.20,
        "TVERSKY_FALSE_POSITIVE_WEIGHT": 0.50,
        "TVERSKY_FALSE_NEGATIVE_WEIGHT": 0.50,
        "TVERSKY_GAMMA": 0.75,
    }

    for name, expected in expected_values.items():
        actual = getattr(training, name)
        if isinstance(expected, float):
            matches = almost_equal(float(actual), expected)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Inherited setting {name} is {actual!r}; expected "
                f"{expected!r}. Refusing an uncontrolled experiment."
            )


def read_hard_negative_ids():
    """Load and validate the training-only hard-negative ID list."""

    if not HARD_NEGATIVE_IDS_PATH.is_file():
        raise FileNotFoundError(
            "Hard-negative ID file was not found: "
            f"{HARD_NEGATIVE_IDS_PATH}"
        )

    image_ids = [
        line.strip()
        for line in HARD_NEGATIVE_IDS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(image_ids) != EXPECTED_HARD_NEGATIVES:
        raise ValueError(
            f"Expected {EXPECTED_HARD_NEGATIVES:,} hard-negative IDs, "
            f"found {len(image_ids):,}."
        )
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("The hard-negative ID file contains duplicates.")

    return set(image_ids)


def classify_training_rows(dataset):
    """Verify all mined IDs and split rows into the three sampler groups."""

    if getattr(dataset, "split", None) != "train":
        raise ValueError("Hard-negative sampling may only use split='train'.")
    if len(dataset) != EXPECTED_TRAINING_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_IMAGES:,} training images, "
            f"found {len(dataset):,}."
        )

    hard_ids = read_hard_negative_ids()
    positive_indices = []
    hard_negative_indices = []
    other_negative_indices = []
    matched_hard_ids = set()

    for index, row in enumerate(dataset.rows):
        image_id = str(row["ImageId"]).strip()
        label = int(row["HasPneumothorax"])

        if label == 1:
            if image_id in hard_ids:
                raise ValueError(
                    "A mined hard-negative ID belongs to a positive "
                    f"training row: {image_id}"
                )
            positive_indices.append(index)
        elif label == 0 and image_id in hard_ids:
            hard_negative_indices.append(index)
            matched_hard_ids.add(image_id)
        elif label == 0:
            other_negative_indices.append(index)
        else:
            raise ValueError(
                f"Training label must be 0 or 1, found {label!r}."
            )

    missing_ids = hard_ids - matched_hard_ids
    if missing_ids:
        first_missing = sorted(missing_ids)[0]
        raise ValueError(
            f"{len(missing_ids):,} hard-negative IDs are absent from the "
            f"training negatives. First missing ID: {first_missing}"
        )

    counts = (
        len(positive_indices),
        len(hard_negative_indices),
        len(other_negative_indices),
    )
    expected_counts = (
        EXPECTED_TRAINING_POSITIVES,
        EXPECTED_HARD_NEGATIVES,
        EXPECTED_OTHER_NEGATIVES,
    )
    if counts != expected_counts:
        raise ValueError(
            "Unexpected training group counts. Found "
            f"positive/hard/other={counts}; expected {expected_counts}."
        )

    return (
        positive_indices,
        hard_negative_indices,
        other_negative_indices,
    )


def create_hard_negative_sampler(dataset):
    """Sample 35% positive, 32.5% hard negative, and 32.5% other."""

    (
        positive_indices,
        hard_negative_indices,
        other_negative_indices,
    ) = classify_training_rows(dataset)

    sample_weights = torch.empty(len(dataset), dtype=torch.double)
    sample_weights[positive_indices] = (
        POSITIVE_SAMPLE_FRACTION / len(positive_indices)
    )
    sample_weights[hard_negative_indices] = (
        HARD_NEGATIVE_SAMPLE_FRACTION / len(hard_negative_indices)
    )
    sample_weights[other_negative_indices] = (
        OTHER_NEGATIVE_SAMPLE_FRACTION / len(other_negative_indices)
    )

    if not almost_equal(float(sample_weights.sum().item()), 1.0):
        raise RuntimeError("Sampler weights do not sum to 1.0.")

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    print("Hard-negative sampler verified")
    print(
        "  Training groups: "
        f"{len(positive_indices):,} positive / "
        f"{len(hard_negative_indices):,} hard negative / "
        f"{len(other_negative_indices):,} other negative"
    )
    print(
        "  Expected sampled mixture: 35.0% positive / "
        "32.5% hard negative / 32.5% other negative"
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def validate_source_checkpoint():
    """Refuse any source other than the locked epoch-4 V1 checkpoint."""

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
        if isinstance(expected, float):
            matches = almost_equal(float(actual), expected)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Unexpected source {description}: {actual!r}; "
                f"expected {expected!r}."
            )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Source checkpoint has no model_state_dict.")


def save_v2a_checkpoint(
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
    """Save V2A without overwriting or changing the locked V1 file."""

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
                "training_positive_fraction": (
                    POSITIVE_SAMPLE_FRACTION
                ),
                "hard_negative_sample_fraction": (
                    HARD_NEGATIVE_SAMPLE_FRACTION
                ),
                "other_negative_sample_fraction": (
                    OTHER_NEGATIVE_SAMPLE_FRACTION
                ),
                "hard_negative_fraction_within_negatives": 0.50,
                "hard_negative_images": EXPECTED_HARD_NEGATIVES,
                "hard_negative_ids_path": str(HARD_NEGATIVE_IDS_PATH),
                "source_checkpoint": str(SOURCE_CHECKPOINT_PATH),
                "source_training_stage": EXPECTED_SOURCE_STAGE,
                "source_checkpoint_epoch": EXPECTED_SOURCE_EPOCH,
                "controlled_change": "hard-negative sampling only",
                "augmentation": "paired_conservative_training_only",
                "loss": (
                    "0.45 weighted BCE + 0.35 symmetric positive "
                    "focal Tversky + 0.20 negative-only BCE"
                ),
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


def load_v2a_resume_state(model, optimizer, scheduler, scaler, device):
    """Resume only a checkpoint produced by this exact V2A experiment."""

    checkpoint = training.load_torch_checkpoint(
        training.LAST_CHECKPOINT_PATH,
        device,
    )
    if checkpoint.get("training_stage") != training.TRAINING_STAGE:
        raise ValueError("Resume checkpoint belongs to another stage.")

    configuration = checkpoint.get("configuration", {})
    required = {
        "controlled_change": "hard-negative sampling only",
        "hard_negative_images": EXPECTED_HARD_NEGATIVES,
        "hard_negative_sample_fraction": (
            HARD_NEGATIVE_SAMPLE_FRACTION
        ),
        "other_negative_sample_fraction": (
            OTHER_NEGATIVE_SAMPLE_FRACTION
        ),
        "test_split_used": False,
    }
    for name, expected in required.items():
        actual = configuration.get(name)
        if isinstance(expected, float):
            matches = almost_equal(float(actual), expected)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Resume setting {name} is {actual!r}; expected "
                f"{expected!r}."
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
    """Apply V2A paths and sampler to the inherited training loop."""

    checkpoint_directory = DRIVE_PROJECT_DIRECTORY / "checkpoints"

    training.SMOKE_TEST = SMOKE_TEST
    training.MAX_EPOCHS = 1 if SMOKE_TEST else 8
    training.MAX_TRAINING_BATCHES = 2 if SMOKE_TEST else None
    training.MAX_VALIDATION_BATCHES = 2 if SMOKE_TEST else None
    training.RESUME_IF_AVAILABLE = not SMOKE_TEST
    training.SOURCE_CHECKPOINT_PATH = SOURCE_CHECKPOINT_PATH
    training.LAST_CHECKPOINT_PATH = checkpoint_directory / (
        "pneumothorax_512_v2a_hard_negative_smoke_last.pth"
        if SMOKE_TEST
        else "pneumothorax_512_v2a_hard_negative_last.pth"
    )
    training.BEST_CHECKPOINT_PATH = checkpoint_directory / (
        "pneumothorax_512_v2a_hard_negative_smoke_best.pth"
        if SMOKE_TEST
        else "pneumothorax_512_v2a_hard_negative_best.pth"
    )
    training.TRAINING_STAGE = (
        "pneumothorax_512_v2a_hard_negative_smoke_test"
        if SMOKE_TEST
        else "pneumothorax_512_v2a_hard_negative_finetune"
    )
    training.create_training_sampler = create_hard_negative_sampler
    training.save_checkpoint = save_v2a_checkpoint
    training.load_resume_state = load_v2a_resume_state


def main():
    validate_unchanged_training_configuration()
    validate_source_checkpoint()
    hard_ids = read_hard_negative_ids()
    configure_experiment()

    print("Pneumothorax Model V2A - hard-negative fine-tuning")
    print("---------------------------------------------------")
    print(f"Run mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL TRAINING'}")
    print(f"Locked V1 source: {SOURCE_CHECKPOINT_PATH}")
    print(f"Mined training hard negatives: {len(hard_ids):,}")
    print("Controlled change: hard-negative sampling only")
    print(
        "Expected sampled mixture: 35.0% positive / "
        "32.5% hard negative / 32.5% other negative"
    )
    print("Validation split: evaluation only")
    print("Test split: not created or accessed\n")

    training.main()


if __name__ == "__main__":
    main()
