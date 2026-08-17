"""Train Pneumothorax Model V3A with stronger positive-overlap pressure.

V3A is a controlled continuation of the locked V1 epoch-4 checkpoint.
It reuses the V1 negative-aware training loop and changes only two loss
coefficients:

    weighted BCE:          0.45 -> 0.25
    positive focal Tversky: 0.35 -> 0.55

The negative-only BCE weight remains 0.20. Architecture, image size, sampler,
augmentation, learning rates, trainable layers, threshold, validation method,
maximum epochs and early stopping remain unchanged. The test split is never
created or accessed.
"""

import math
from pathlib import Path

import torch

import train_pneumothorax_512_negative_aware as training


RANDOM_SEED = 42

EXPECTED_SOURCE_STAGE = "pneumothorax_512_negative_aware_finetune"
EXPECTED_SOURCE_EPOCH = 4
EXPECTED_IMAGE_SIZE = 512
EXPECTED_THRESHOLD = 0.35

V1_BCE_WEIGHT = 0.45
V1_POSITIVE_TVERSKY_WEIGHT = 0.35
V1_NEGATIVE_BCE_WEIGHT = 0.20

V3A_BCE_WEIGHT = 0.25
V3A_POSITIVE_TVERSKY_WEIGHT = 0.55
V3A_NEGATIVE_BCE_WEIGHT = 0.20

DRIVE_PROJECT_DIRECTORY = Path(
    "/content/drive/MyDrive/SmartMedicalImagingColab"
)
SOURCE_CHECKPOINT_PATH = (
    DRIVE_PROJECT_DIRECTORY
    / "checkpoints"
    / "pneumothorax_512_negative_aware_best.pth"
)

# Keep True for the first two-batch Colab smoke test. After it passes, change
# only this value to False and rerun the script for full V3A training.
SMOKE_TEST = True


def almost_equal(first, second):
    return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1e-12)


def validate_inherited_v1_configuration():
    """Refuse to run if the inherited V1 recipe has unexpectedly changed."""

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
        "BCE_WEIGHT": V1_BCE_WEIGHT,
        "POSITIVE_TVERSKY_WEIGHT": V1_POSITIVE_TVERSKY_WEIGHT,
        "NEGATIVE_BCE_WEIGHT": V1_NEGATIVE_BCE_WEIGHT,
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


def save_v3a_checkpoint(
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
    """Save V3A state without overwriting V1 or V2A checkpoints."""

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
                "training_positive_fraction": (
                    training.TRAINING_POSITIVE_FRACTION
                ),
                "source_checkpoint": str(SOURCE_CHECKPOINT_PATH),
                "source_training_stage": EXPECTED_SOURCE_STAGE,
                "source_checkpoint_epoch": EXPECTED_SOURCE_EPOCH,
                "controlled_change": "positive-overlap loss balance only",
                "augmentation": "paired_conservative_training_only",
                "loss": (
                    "0.25 weighted BCE + 0.55 symmetric positive "
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


def load_v3a_resume_state(model, optimizer, scheduler, scaler, device):
    """Resume only a checkpoint produced by this exact V3A experiment."""

    checkpoint = training.load_torch_checkpoint(
        training.LAST_CHECKPOINT_PATH,
        device,
    )

    if checkpoint.get("training_stage") != training.TRAINING_STAGE:
        raise ValueError("Resume checkpoint belongs to another training stage.")

    configuration = checkpoint.get("configuration", {})
    required = {
        "controlled_change": "positive-overlap loss balance only",
        "source_training_stage": EXPECTED_SOURCE_STAGE,
        "source_checkpoint_epoch": EXPECTED_SOURCE_EPOCH,
        "bce_weight": V3A_BCE_WEIGHT,
        "positive_tversky_weight": V3A_POSITIVE_TVERSKY_WEIGHT,
        "negative_bce_weight": V3A_NEGATIVE_BCE_WEIGHT,
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
            f"V3A resume checkpoint is missing: {sorted(missing)}"
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
    """Apply V3A loss, paths and safety metadata to the V1 loop."""

    checkpoint_directory = DRIVE_PROJECT_DIRECTORY / "checkpoints"

    training.BCE_WEIGHT = V3A_BCE_WEIGHT
    training.POSITIVE_TVERSKY_WEIGHT = V3A_POSITIVE_TVERSKY_WEIGHT
    training.NEGATIVE_BCE_WEIGHT = V3A_NEGATIVE_BCE_WEIGHT

    training.SMOKE_TEST = SMOKE_TEST
    training.MAX_EPOCHS = 1 if SMOKE_TEST else 8
    training.MAX_TRAINING_BATCHES = 2 if SMOKE_TEST else None
    training.MAX_VALIDATION_BATCHES = 2 if SMOKE_TEST else None
    training.RESUME_IF_AVAILABLE = not SMOKE_TEST

    training.SOURCE_CHECKPOINT_PATH = SOURCE_CHECKPOINT_PATH
    training.LAST_CHECKPOINT_PATH = checkpoint_directory / (
        "pneumothorax_512_v3a_positive_overlap_smoke_last.pth"
        if SMOKE_TEST
        else "pneumothorax_512_v3a_positive_overlap_last.pth"
    )
    training.BEST_CHECKPOINT_PATH = checkpoint_directory / (
        "pneumothorax_512_v3a_positive_overlap_smoke_best.pth"
        if SMOKE_TEST
        else "pneumothorax_512_v3a_positive_overlap_best.pth"
    )
    training.TRAINING_STAGE = (
        "pneumothorax_512_v3a_positive_overlap_smoke_test"
        if SMOKE_TEST
        else "pneumothorax_512_v3a_positive_overlap_finetune"
    )

    training.save_checkpoint = save_v3a_checkpoint
    training.load_resume_state = load_v3a_resume_state


def main():
    validate_inherited_v1_configuration()
    validate_source_checkpoint()
    configure_experiment()

    print("Pneumothorax Model V3A - positive-overlap loss experiment")
    print("----------------------------------------------------------")
    print(f"Run mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL TRAINING'}")
    print(f"Locked V1 source: {SOURCE_CHECKPOINT_PATH}")
    print("Controlled change: loss balance only")
    print("  Weighted BCE: 0.45 -> 0.25")
    print("  Positive focal Tversky: 0.35 -> 0.55")
    print("  Negative-only BCE: unchanged at 0.20")
    print("Validation split: checkpoint selection only")
    print("Test split: not created or accessed\n")

    training.main()


if __name__ == "__main__":
    main()
