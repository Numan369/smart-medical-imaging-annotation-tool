import csv
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pneumothorax_dataset import PneumothoraxDataset
from pneumothorax_model import PneumothoraxResNet34UNet


CHECKPOINT_PATH = Path("checkpoints") / "fine_tune_stage_best.pth"
EXPECTED_TRAINING_STAGE = (
    "balanced_weighted_partial_encoder_finetune"
)
PREDICTION_THRESHOLD = 0.35
IMAGE_SIZE = 256
BATCH_SIZE = 4
PROGRESS_INTERVAL = 50

OUTPUT_DIRECTORY = Path("validation_analysis")
DETAILS_PATH = OUTPUT_DIRECTORY / "positive_case_details.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "failure_analysis_summary.txt"


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_checkpoint(model, device):
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "Checkpoint was not found at: "
            f"{CHECKPOINT_PATH.resolve()}"
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

    stage = checkpoint.get("training_stage")

    if stage != EXPECTED_TRAINING_STAGE:
        raise ValueError(
            "Unexpected checkpoint training stage: "
            f"{stage!r}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def mask_size_group(area_percentage):
    """Group positive masks by their share of the resized X-ray."""

    if area_percentage <= 0.10:
        return "tiny (<=0.10%)"
    if area_percentage <= 0.50:
        return "small (0.10-0.50%)"
    if area_percentage <= 2.00:
        return "medium (0.50-2.00%)"
    return "large (>2.00%)"


def safe_mean(values):
    if not values:
        return float("nan")

    return sum(values) / len(values)


@torch.no_grad()
def analyse_validation(model, data_loader, device):
    model.eval()

    rows = []
    total_batches = len(data_loader)
    start_time = time.perf_counter()

    for batch_number, batch in enumerate(data_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True) >= 0.5
        image_ids = batch["image_id"]

        probabilities = torch.sigmoid(model(images))
        predictions = probabilities >= PREDICTION_THRESHOLD

        if predictions.shape != targets.shape:
            raise ValueError(
                "Prediction and target-mask shapes do not match."
            )

        for index, image_id in enumerate(image_ids):
            target = targets[index]

            # Failure analysis intentionally includes positive validation
            # images only. Negative-image performance was already measured
            # during validation threshold tuning.
            if not target.any():
                continue

            probability = probabilities[index]
            prediction = predictions[index]

            target_pixels = int(target.sum().item())
            predicted_pixels = int(prediction.sum().item())
            intersection = int((prediction & target).sum().item())
            denominator = target_pixels + predicted_pixels
            dice = (
                2.0 * intersection / denominator
                if denominator > 0
                else 1.0
            )

            image_pixels = target.numel()
            target_area_percentage = 100.0 * target_pixels / image_pixels
            predicted_area_percentage = (
                100.0 * predicted_pixels / image_pixels
            )

            target_probabilities = probability[target]
            background_probabilities = probability[~target]

            rows.append(
                {
                    "image_id": image_id,
                    "size_group": mask_size_group(
                        target_area_percentage
                    ),
                    "target_pixels": target_pixels,
                    "target_area_percent": target_area_percentage,
                    "predicted_pixels": predicted_pixels,
                    "predicted_area_percent": predicted_area_percentage,
                    "intersection_pixels": intersection,
                    "dice": dice,
                    "maximum_probability": float(
                        probability.max().item()
                    ),
                    "mean_probability_inside_target": float(
                        target_probabilities.mean().item()
                    ),
                    "mean_probability_outside_target": float(
                        background_probabilities.mean().item()
                    ),
                    "empty_prediction": int(predicted_pixels == 0),
                }
            )

        if (
            batch_number == 1
            or batch_number % PROGRESS_INTERVAL == 0
            or batch_number == total_batches
        ):
            elapsed_minutes = (
                time.perf_counter() - start_time
            ) / 60.0
            print(
                f"Validation [{batch_number}/{total_batches}] "
                f"elapsed={elapsed_minutes:.1f} min"
            )

    return rows, time.perf_counter() - start_time


def build_summary(rows, elapsed_seconds):
    if not rows:
        raise ValueError(
            "No positive validation images were analysed."
        )

    empty_rows = [row for row in rows if row["empty_prediction"]]
    grouped_rows = defaultdict(list)

    for row in rows:
        grouped_rows[row["size_group"]].append(row)

    lines = [
        "Validation failure analysis",
        "---------------------------",
        f"Checkpoint: {CHECKPOINT_PATH.resolve()}",
        f"Prediction threshold: {PREDICTION_THRESHOLD:.2f}",
        f"Model input size: {IMAGE_SIZE} x {IMAGE_SIZE}",
        "Dataset split: validation only (test is not used)",
        "",
        f"Positive validation images: {len(rows)}",
        (
            "Mean positive-case Dice: "
            f"{safe_mean([row['dice'] for row in rows]):.6f}"
        ),
        (
            "Completely empty predictions: "
            f"{len(empty_rows)} / {len(rows)} "
            f"({100.0 * len(empty_rows) / len(rows):.2f}%)"
        ),
        (
            "Mean maximum probability on missed cases: "
            f"{safe_mean([row['maximum_probability'] for row in empty_rows]):.6f}"
        ),
        "",
        "Results by expert-mask size",
        "Size group | Cases | Empty predictions | Miss rate | Mean Dice",
    ]

    group_order = [
        "tiny (<=0.10%)",
        "small (0.10-0.50%)",
        "medium (0.50-2.00%)",
        "large (>2.00%)",
    ]

    for group_name in group_order:
        group = grouped_rows.get(group_name, [])

        if not group:
            continue

        empty_count = sum(row["empty_prediction"] for row in group)
        miss_rate = 100.0 * empty_count / len(group)
        mean_dice = safe_mean([row["dice"] for row in group])
        lines.append(
            f"{group_name} | {len(group)} | {empty_count} | "
            f"{miss_rate:.2f}% | {mean_dice:.6f}"
        )

    hardest = sorted(
        rows,
        key=lambda row: (
            row["dice"],
            row["target_area_percent"],
        ),
    )[:10]

    lines.extend(
        [
            "",
            "Ten hardest positive validation images",
            "Image ID | Size group | Target area | Max probability | Dice",
        ]
    )

    for row in hardest:
        lines.append(
            f"{row['image_id']} | {row['size_group']} | "
            f"{row['target_area_percent']:.4f}% | "
            f"{row['maximum_probability']:.4f} | "
            f"{row['dice']:.6f}"
        )

    lines.extend(
        [
            "",
            f"Analysis time: {elapsed_seconds / 60.0:.1f} min",
            "No model parameters or checkpoint files were changed.",
        ]
    )

    return "\n".join(lines)


def save_details(rows):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with DETAILS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    device = choose_device()

    print("Validation-set pneumothorax failure analysis")
    print("--------------------------------------------")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print(f"Prediction threshold: {PREDICTION_THRESHOLD:.2f}")
    print("Dataset split: validation only (test is not used)")

    validation_dataset = PneumothoraxDataset(
        split="validation",
        image_size=IMAGE_SIZE,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    print(f"Validation images: {len(validation_dataset)}")
    print(f"Validation batches: {len(validation_loader)}")
    print("Loading model and checkpoint...")

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=False,
    ).to(device)
    checkpoint = load_checkpoint(model, device)
    print(
        "Checkpoint epoch: "
        f"{checkpoint.get('completed_epoch', 'unknown')}"
    )

    print("\nAnalysing positive validation cases...")
    rows, elapsed_seconds = analyse_validation(
        model,
        validation_loader,
        device,
    )

    save_details(rows)
    summary = build_summary(rows, elapsed_seconds)
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")

    print("\n## Validation failure-analysis results\n")
    print(summary)
    print(f"\nDetailed CSV: {DETAILS_PATH.resolve()}")
    print(f"Summary file: {SUMMARY_PATH.resolve()}")


if __name__ == "__main__":
    main()
