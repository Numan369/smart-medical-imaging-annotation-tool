"""Mine hard negatives for pneumothorax Model V2.

This script runs the locked V1 checkpoint over negative images from the
training split only. It never opens the validation or test splits and never
modifies the V1 checkpoint. Negative images that receive a non-empty mask at
the fixed V1 threshold are recorded as hard negatives for later V2 sampling.
"""

import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from pneumothorax_dataset import PneumothoraxDataset
from pneumothorax_model import PneumothoraxResNet34UNet


DRIVE_PROJECT_DIRECTORY = Path(
    "/content/drive/MyDrive/SmartMedicalImagingColab"
)
CHECKPOINT_PATH = (
    DRIVE_PROJECT_DIRECTORY
    / "checkpoints"
    / "pneumothorax_512_negative_aware_best.pth"
)
OUTPUT_DIRECTORY = DRIVE_PROJECT_DIRECTORY / "hard_negative_mining_v2"
DETAILS_PATH = OUTPUT_DIRECTORY / "training_negative_predictions.csv"
HARD_NEGATIVES_PATH = OUTPUT_DIRECTORY / "hard_negative_ids.txt"
SUMMARY_PATH = OUTPUT_DIRECTORY / "hard_negative_summary.json"

EXPECTED_TRAINING_STAGE = "pneumothorax_512_negative_aware_finetune"
EXPECTED_COMPLETED_EPOCH = 4
EXPECTED_IMAGE_SIZE = 512
EXPECTED_PREDICTION_THRESHOLD = 0.35
EXPECTED_TRAINING_IMAGES = 9637
EXPECTED_TRAINING_NEGATIVES = 7502

BATCH_SIZE = 8
PROGRESS_INTERVAL = 100


def choose_device():
    """Use a Colab GPU when available."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_torch_checkpoint(path, device):
    """Load checkpoints on both older and newer PyTorch versions."""

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def validate_checkpoint(checkpoint):
    """Refuse to mine with anything other than the locked V1 model."""

    stage = checkpoint.get("training_stage")
    if stage != EXPECTED_TRAINING_STAGE:
        raise ValueError(
            f"Unexpected training stage {stage!r}; expected "
            f"{EXPECTED_TRAINING_STAGE!r}."
        )

    epoch = checkpoint.get("completed_epoch")
    if epoch != EXPECTED_COMPLETED_EPOCH:
        raise ValueError(
            f"Unexpected checkpoint epoch {epoch!r}; expected epoch "
            f"{EXPECTED_COMPLETED_EPOCH}."
        )

    configuration = checkpoint.get("configuration", {})
    if configuration.get("test_split_used") is not False:
        raise ValueError(
            "Checkpoint metadata does not confirm an untouched test split."
        )

    image_size = int(configuration.get("image_size", -1))
    threshold = float(configuration.get("prediction_threshold", -1.0))

    if image_size != EXPECTED_IMAGE_SIZE:
        raise ValueError(
            f"Unexpected image size {image_size}; expected "
            f"{EXPECTED_IMAGE_SIZE}."
        )

    if not math.isclose(
        threshold,
        EXPECTED_PREDICTION_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Unexpected prediction threshold {threshold}; expected "
            f"{EXPECTED_PREDICTION_THRESHOLD}."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")

    return image_size, threshold


def get_training_negative_indices(dataset):
    """Select negative rows without reading any DICOM pixel data."""

    if len(dataset) != EXPECTED_TRAINING_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_IMAGES:,} training images, found "
            f"{len(dataset):,}."
        )

    indices = [
        index
        for index, row in enumerate(dataset.rows)
        if int(row["HasPneumothorax"]) == 0
    ]

    if len(indices) != EXPECTED_TRAINING_NEGATIVES:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_NEGATIVES:,} training negatives, "
            f"found {len(indices):,}."
        )

    return indices


@torch.inference_mode()
def mine_hard_negatives(model, loader, device, threshold, image_size):
    """Measure every training negative and rank the V1 false alarms."""

    model.eval()
    rows = []
    total_batches = len(loader)
    pixels_per_image = image_size * image_size
    use_amp = device.type == "cuda"
    started = time.perf_counter()

    for batch_number, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"]

        if torch.any(targets >= 0.5):
            raise ValueError(
                "A non-empty mask appeared in the negative-only mining set."
            )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            probabilities = torch.sigmoid(model(images))

        predictions = probabilities >= threshold
        predicted_pixels = predictions.flatten(1).sum(dim=1)
        maximum_probabilities = probabilities.flatten(1).amax(dim=1)
        mean_probabilities = probabilities.flatten(1).mean(dim=1)

        for index, image_id in enumerate(batch["image_id"]):
            pixel_count = int(predicted_pixels[index].item())
            rows.append(
                {
                    "image_id": str(image_id),
                    "predicted_pixels": pixel_count,
                    "predicted_area_percent": (
                        100.0 * pixel_count / pixels_per_image
                    ),
                    "maximum_probability": float(
                        maximum_probabilities[index].item()
                    ),
                    "mean_probability": float(
                        mean_probabilities[index].item()
                    ),
                    "is_hard_negative": int(pixel_count > 0),
                }
            )

        if (
            batch_number == 1
            or batch_number % PROGRESS_INTERVAL == 0
            or batch_number == total_batches
        ):
            elapsed = time.perf_counter() - started
            print(
                f"Batch {batch_number:,}/{total_batches:,} | "
                f"processed {len(rows):,}/{EXPECTED_TRAINING_NEGATIVES:,} "
                f"negatives | {elapsed / 60.0:.1f} min"
            )

    if len(rows) != EXPECTED_TRAINING_NEGATIVES:
        raise RuntimeError(
            f"Mining processed {len(rows):,} images; expected "
            f"{EXPECTED_TRAINING_NEGATIVES:,}."
        )

    rows.sort(
        key=lambda row: (
            row["predicted_pixels"],
            row["maximum_probability"],
        ),
        reverse=True,
    )

    for rank, row in enumerate(rows, start=1):
        row["hardness_rank"] = rank if row["is_hard_negative"] else ""

    return rows, time.perf_counter() - started


def save_results(rows, elapsed_seconds, device):
    """Save full measurements, hard-negative IDs, and an audit summary."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "predicted_pixels",
        "predicted_area_percent",
        "maximum_probability",
        "mean_probability",
        "is_hard_negative",
        "hardness_rank",
    ]

    with DETAILS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    hard_rows = [row for row in rows if row["is_hard_negative"]]
    HARD_NEGATIVES_PATH.write_text(
        "\n".join(row["image_id"] for row in hard_rows) + "\n",
        encoding="utf-8",
    )

    summary = {
        "purpose": "Model V2 hard-negative mining",
        "source_checkpoint": str(CHECKPOINT_PATH),
        "source_training_stage": EXPECTED_TRAINING_STAGE,
        "source_checkpoint_epoch": EXPECTED_COMPLETED_EPOCH,
        "dataset_split_used": "train",
        "validation_split_used": False,
        "test_split_used": False,
        "prediction_threshold": EXPECTED_PREDICTION_THRESHOLD,
        "image_size": EXPECTED_IMAGE_SIZE,
        "training_negative_images": len(rows),
        "hard_negative_images": len(hard_rows),
        "hard_negative_rate": len(hard_rows) / len(rows),
        "largest_false_positive_pixels": (
            int(hard_rows[0]["predicted_pixels"]) if hard_rows else 0
        ),
        "largest_false_positive_area_percent": (
            float(hard_rows[0]["predicted_area_percent"])
            if hard_rows
            else 0.0
        ),
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    print("Pneumothorax Model V2 - hard-negative mining")
    print("------------------------------------------------")
    print("Dataset split: training negatives only")
    print("Validation split: not created or accessed")
    print("Test split: not created or accessed")

    device = choose_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Locked V1 checkpoint was not found: {CHECKPOINT_PATH}"
        )

    checkpoint = load_torch_checkpoint(CHECKPOINT_PATH, device)
    image_size, threshold = validate_checkpoint(checkpoint)
    print(f"V1 checkpoint: {CHECKPOINT_PATH}")
    print(f"Checkpoint epoch: {checkpoint['completed_epoch']}")
    print(f"Image size: {image_size} x {image_size}")
    print(f"Fixed V1 threshold: {threshold}")

    training_dataset = PneumothoraxDataset(
        split="train",
        image_size=image_size,
    )
    negative_indices = get_training_negative_indices(training_dataset)
    negative_dataset = Subset(training_dataset, negative_indices)

    number_of_workers = 2 if device.type == "cuda" else 0
    loader = DataLoader(
        negative_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=number_of_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    print(f"Training negatives: {len(negative_dataset):,}")
    print(f"Mining batches: {len(loader):,}")

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    rows, elapsed_seconds = mine_hard_negatives(
        model=model,
        loader=loader,
        device=device,
        threshold=threshold,
        image_size=image_size,
    )
    summary = save_results(rows, elapsed_seconds, device)

    print("\nHard-negative mining complete")
    print(
        "Hard negatives: "
        f"{summary['hard_negative_images']:,}/"
        f"{summary['training_negative_images']:,} "
        f"({100.0 * summary['hard_negative_rate']:.2f}%)"
    )
    print(f"Details CSV: {DETAILS_PATH}")
    print(f"Hard-negative IDs: {HARD_NEGATIVES_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("Validation and test splits were not used.")


if __name__ == "__main__":
    main()
