"""Validation-only diagnosis of the selected V3B epoch-3 checkpoint.

This guarded wrapper reuses the established V1 diagnostic implementation but
points it at the V3B best checkpoint and a separate output directory. It
validates checkpoint identity before inference. It never creates the test split,
trains, performs backpropagation, or modifies a checkpoint.
"""

from pathlib import Path

import torch

import diagnose_pneumothorax_512_negative_aware_validation as diagnostic
from pneumothorax_model import PneumothoraxResNet34UNet


EXPECTED_TRAINING_STAGE = (
    "pneumothorax_512_v3b_small_lesion_sampling_finetune"
)
EXPECTED_COMPLETED_EPOCH = 3
EXPECTED_IMAGE_SIZE = 512
EXPECTED_THRESHOLD = 0.35
EXPECTED_CONTROLLED_CHANGE = (
    "size-aware sampling within positive images only"
)

DRIVE_PROJECT_DIRECTORY = Path(
    "/content/drive/MyDrive/SmartMedicalImagingColab"
)
CHECKPOINT_PATH = (
    DRIVE_PROJECT_DIRECTORY
    / "checkpoints"
    / "pneumothorax_512_v3b_small_lesion_sampling_best.pth"
)
OUTPUT_DIRECTORY = (
    DRIVE_PROJECT_DIRECTORY
    / "diagnostics_v3b_small_lesion_validation"
)


def load_v3b_model(device):
    """Load only the validation-selected V3B epoch-3 checkpoint."""

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"V3B best checkpoint was not found: {CHECKPOINT_PATH}"
        )

    checkpoint = diagnostic.load_torch_checkpoint(
        CHECKPOINT_PATH,
        device,
    )
    configuration = checkpoint.get("configuration", {})

    checks = {
        "training stage": (
            checkpoint.get("training_stage"),
            EXPECTED_TRAINING_STAGE,
        ),
        "completed epoch": (
            checkpoint.get("completed_epoch"),
            EXPECTED_COMPLETED_EPOCH,
        ),
        "image size": (
            configuration.get("image_size"),
            EXPECTED_IMAGE_SIZE,
        ),
        "prediction threshold": (
            configuration.get("prediction_threshold"),
            EXPECTED_THRESHOLD,
        ),
        "controlled change": (
            configuration.get("controlled_change"),
            EXPECTED_CONTROLLED_CHANGE,
        ),
        "validation split used": (
            configuration.get("validation_split_used"),
            True,
        ),
        "test split used": (
            configuration.get("test_split_used"),
            False,
        ),
    }

    for description, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"Unexpected V3B {description}: {actual!r}; "
                f"expected {expected!r}."
            )

    if "model_state_dict" not in checkpoint:
        raise KeyError("V3B checkpoint has no model_state_dict.")

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        "Loaded V3B checkpoint epoch "
        f"{checkpoint['completed_epoch']} "
        f"({checkpoint['training_stage']})"
    )
    return model, checkpoint


def configure_diagnostic():
    """Use V3B paths while preserving the established validation procedure."""

    diagnostic.CHECKPOINT_PATH = CHECKPOINT_PATH
    diagnostic.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
    diagnostic.DEPLOYED_THRESHOLD = EXPECTED_THRESHOLD
    diagnostic.IMAGE_SIZE = EXPECTED_IMAGE_SIZE
    diagnostic.BATCH_SIZE = 2
    diagnostic.load_locked_model = load_v3b_model


def main():
    print("V3B validation-only lesion-size diagnosis")
    print("-----------------------------------------")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output: {OUTPUT_DIRECTORY}")
    print("Validation split only; test split will not be instantiated.")
    print("No training or checkpoint modification.\n")

    configure_diagnostic()
    diagnostic.main()


if __name__ == "__main__":
    main()
