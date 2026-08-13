import csv
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pneumothorax_dataset import PneumothoraxDataset
from pneumothorax_model import PneumothoraxResNet34UNet


IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_WORKERS = 0
PROGRESS_INTERVAL = 50
PREDICTION_THRESHOLD = 0.35
MINIMUM_AREA_CANDIDATES = (0, 16, 32, 64, 128, 256, 512, 1024)

CHECKPOINT_PATH = Path("checkpoints") / "pneumothorax_512_best.pth"
EXPECTED_TRAINING_STAGE = "pneumothorax_512_augmented_progressive_finetune"
OUTPUT_DIRECTORY = Path("validation_analysis") / "errors_512"
POSITIVE_DETAILS_PATH = OUTPUT_DIRECTORY / "positive_case_details.csv"
NEGATIVE_DETAILS_PATH = OUTPUT_DIRECTORY / "negative_case_details.csv"
AREA_RESULTS_PATH = OUTPUT_DIRECTORY / "minimum_area_results.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "error_analysis_summary.txt"


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, device):
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found: {CHECKPOINT_PATH.resolve()}"
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")

    stage = checkpoint.get("training_stage")
    if stage != EXPECTED_TRAINING_STAGE:
        raise ValueError(
            f"Unexpected training stage {stage!r}; expected "
            f"{EXPECTED_TRAINING_STAGE!r}."
        )

    configuration = checkpoint.get("configuration", {})
    checkpoint_size = configuration.get("image_size")
    if checkpoint_size not in (None, IMAGE_SIZE):
        raise ValueError(
            f"Checkpoint image size is {checkpoint_size!r}, not 512."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def mask_size_group(area_percentage):
    if area_percentage <= 0.10:
        return "tiny (<=0.10%)"
    if area_percentage <= 0.50:
        return "small (0.10-0.50%)"
    if area_percentage <= 2.00:
        return "medium (0.50-2.00%)"
    return "large (>2.00%)"


def safe_mean(values):
    return sum(values) / len(values) if values else float("nan")


def percentile(values, percentage):
    if not values:
        return float("nan")

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@torch.inference_mode()
def analyse_validation(model, loader, device):
    model.eval()
    positive_rows = []
    negative_rows = []
    total_batches = len(loader)
    start_time = time.perf_counter()

    for batch_number, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True) >= 0.5
        probabilities = torch.sigmoid(model(images))
        predictions = probabilities >= PREDICTION_THRESHOLD

        if predictions.shape != targets.shape:
            raise ValueError("Prediction and target shapes do not match.")

        for index, image_id in enumerate(batch["image_id"]):
            target = targets[index]
            probability = probabilities[index]
            prediction = predictions[index]
            target_pixels = int(target.sum().item())
            predicted_pixels = int(prediction.sum().item())
            maximum_probability = float(probability.max().item())

            if target_pixels == 0:
                negative_rows.append(
                    {
                        "image_id": image_id,
                        "predicted_pixels": predicted_pixels,
                        "predicted_area_percent": (
                            100.0 * predicted_pixels / target.numel()
                        ),
                        "maximum_probability": maximum_probability,
                        "false_positive": int(predicted_pixels > 0),
                    }
                )
                continue

            intersection = int((prediction & target).sum().item())
            denominator = target_pixels + predicted_pixels
            dice = 2.0 * intersection / denominator if denominator else 1.0
            area_percentage = 100.0 * target_pixels / target.numel()

            positive_rows.append(
                {
                    "image_id": image_id,
                    "size_group": mask_size_group(area_percentage),
                    "target_pixels": target_pixels,
                    "target_area_percent": area_percentage,
                    "predicted_pixels": predicted_pixels,
                    "predicted_area_percent": (
                        100.0 * predicted_pixels / target.numel()
                    ),
                    "intersection_pixels": intersection,
                    "dice": dice,
                    "maximum_probability": maximum_probability,
                    "empty_prediction": int(predicted_pixels == 0),
                }
            )

        if (
            batch_number == 1
            or batch_number % PROGRESS_INTERVAL == 0
            or batch_number == total_batches
        ):
            elapsed = (time.perf_counter() - start_time) / 60.0
            print(
                f"Validation [{batch_number}/{total_batches}] "
                f"elapsed={elapsed:.1f} min"
            )

    return positive_rows, negative_rows, time.perf_counter() - start_time


def calculate_area_results(positive_rows, negative_rows):
    results = []

    for minimum_area in MINIMUM_AREA_CANDIDATES:
        dice_values = []
        empty_positive_count = 0

        for row in positive_rows:
            keep = row["predicted_pixels"] >= minimum_area
            if not keep:
                dice_values.append(0.0)
                empty_positive_count += 1
            else:
                dice_values.append(row["dice"])
                empty_positive_count += row["empty_prediction"]

        false_positive_count = sum(
            row["predicted_pixels"] >= max(1, minimum_area)
            for row in negative_rows
        )
        positive_dice = safe_mean(dice_values)
        negative_empty_accuracy = 1.0 - (
            false_positive_count / len(negative_rows)
        )
        balanced_score = (
            2.0
            * positive_dice
            * negative_empty_accuracy
            / (positive_dice + negative_empty_accuracy)
            if positive_dice + negative_empty_accuracy > 0
            else 0.0
        )

        results.append(
            {
                "minimum_predicted_pixels": minimum_area,
                "positive_case_dice": positive_dice,
                "empty_positive_predictions": empty_positive_count,
                "empty_positive_rate": (
                    empty_positive_count / len(positive_rows)
                ),
                "false_positive_negative_images": false_positive_count,
                "false_positive_negative_rate": (
                    false_positive_count / len(negative_rows)
                ),
                "negative_empty_accuracy": negative_empty_accuracy,
                "balanced_score": balanced_score,
            }
        )

    return results


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(positive_rows, negative_rows, area_results, elapsed):
    grouped = defaultdict(list)
    for row in positive_rows:
        grouped[row["size_group"]].append(row)

    false_positive_rows = [
        row for row in negative_rows if row["false_positive"]
    ]
    negative_areas = [row["predicted_pixels"] for row in negative_rows]
    best_area = max(
        area_results,
        key=lambda row: (
            row["balanced_score"],
            row["positive_case_dice"],
            -row["minimum_predicted_pixels"],
        ),
    )

    lines = [
        "Pneumothorax 512 x 512 validation error analysis",
        "==================================================",
        f"Checkpoint: {CHECKPOINT_PATH.resolve()}",
        f"Probability threshold: {PREDICTION_THRESHOLD:.2f} (provisional)",
        f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}",
        "Dataset split: validation only",
        "Test split used: False",
        "Training performed: False",
        "",
        f"Positive images: {len(positive_rows)}",
        f"Negative images: {len(negative_rows)}",
        f"Positive-case Dice: "
        f"{safe_mean([row['dice'] for row in positive_rows]):.6f}",
        f"Empty positive predictions: "
        f"{sum(row['empty_prediction'] for row in positive_rows)} / "
        f"{len(positive_rows)}",
        f"False-positive negative images: {len(false_positive_rows)} / "
        f"{len(negative_rows)} "
        f"({100.0 * len(false_positive_rows) / len(negative_rows):.2f}%)",
        "",
        "Positive results by expert-mask size",
        "Size group | Cases | Empty | Miss rate | Mean Dice",
    ]

    group_order = [
        "tiny (<=0.10%)",
        "small (0.10-0.50%)",
        "medium (0.50-2.00%)",
        "large (>2.00%)",
    ]
    for group_name in group_order:
        rows = grouped.get(group_name, [])
        if not rows:
            continue
        empty = sum(row["empty_prediction"] for row in rows)
        lines.append(
            f"{group_name} | {len(rows)} | {empty} | "
            f"{100.0 * empty / len(rows):.2f}% | "
            f"{safe_mean([row['dice'] for row in rows]):.6f}"
        )

    lines.extend(
        [
            "",
            "Predicted area on negative images at threshold 0.35",
            f"Median pixels: {percentile(negative_areas, 50):.1f}",
            f"75th percentile: {percentile(negative_areas, 75):.1f}",
            f"90th percentile: {percentile(negative_areas, 90):.1f}",
            f"95th percentile: {percentile(negative_areas, 95):.1f}",
            f"Maximum pixels: {max(negative_areas)}",
            "",
            "Minimum predicted-area experiment",
            "Min pixels | Pos Dice | Empty positives | FP negatives | Balanced",
        ]
    )

    for row in area_results:
        marker = " < best diagnostic balance" if row is best_area else ""
        lines.append(
            f"{row['minimum_predicted_pixels']:10d} | "
            f"{row['positive_case_dice']:.6f} | "
            f"{row['empty_positive_predictions']:3d}/"
            f"{100.0 * row['empty_positive_rate']:.2f}% | "
            f"{row['false_positive_negative_images']:3d}/"
            f"{100.0 * row['false_positive_negative_rate']:.2f}% | "
            f"{row['balanced_score']:.6f}{marker}"
        )

    hardest = sorted(
        positive_rows,
        key=lambda row: (row["dice"], row["target_area_percent"]),
    )[:10]
    strongest_false_positives = sorted(
        false_positive_rows,
        key=lambda row: (
            row["predicted_pixels"], row["maximum_probability"]
        ),
        reverse=True,
    )[:10]

    lines.extend(
        [
            "",
            "Ten hardest positive images",
            "Image ID | Size group | Target area | Predicted pixels | Dice",
        ]
    )
    for row in hardest:
        lines.append(
            f"{row['image_id']} | {row['size_group']} | "
            f"{row['target_area_percent']:.4f}% | "
            f"{row['predicted_pixels']} | {row['dice']:.6f}"
        )

    lines.extend(
        [
            "",
            "Ten largest false-positive predictions",
            "Image ID | Predicted pixels | Predicted area | Max probability",
        ]
    )
    for row in strongest_false_positives:
        lines.append(
            f"{row['image_id']} | {row['predicted_pixels']} | "
            f"{row['predicted_area_percent']:.4f}% | "
            f"{row['maximum_probability']:.6f}"
        )

    lines.extend(
        [
            "",
            f"Evaluation time: {elapsed / 60.0:.1f} min",
            "Area filtering was analysed only; no final rule was selected.",
            "No checkpoint or model parameters were changed.",
        ]
    )
    return "\n".join(lines)


def print_area_table(area_results):
    print("\nMinimum predicted-area diagnostic")
    print("Min px | Pos Dice | Empty pos | FP negatives | Balanced")
    for row in area_results:
        print(
            f"{row['minimum_predicted_pixels']:6d} | "
            f"{row['positive_case_dice']:.6f} | "
            f"{row['empty_positive_predictions']:3d}/"
            f"{100.0 * row['empty_positive_rate']:5.2f}% | "
            f"{row['false_positive_negative_images']:3d}/"
            f"{100.0 * row['false_positive_negative_rate']:5.2f}% | "
            f"{row['balanced_score']:.6f}"
        )


def main():
    device = choose_device()
    print("Pneumothorax 512 x 512 validation error analysis")
    print("------------------------------------------------")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print(f"Probability threshold: {PREDICTION_THRESHOLD:.2f} (provisional)")
    print("Dataset split: validation only")
    print("Test split: not created or accessed")
    print("Training: disabled")

    dataset = PneumothoraxDataset(split="validation", image_size=IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    print(f"Validation images: {len(dataset):,}")
    print(f"Validation batches: {len(loader):,}")

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False, freeze_encoder=False
    ).to(device)
    checkpoint = load_checkpoint(model, device)
    print(f"Checkpoint epoch: {checkpoint.get('completed_epoch', 'unknown')}")
    print("\nAnalysing validation errors...")

    positive_rows, negative_rows, elapsed = analyse_validation(
        model, loader, device
    )
    if not positive_rows or not negative_rows:
        raise ValueError("Validation must contain positive and negative images.")

    area_results = calculate_area_results(positive_rows, negative_rows)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_csv(POSITIVE_DETAILS_PATH, positive_rows)
    write_csv(NEGATIVE_DETAILS_PATH, negative_rows)
    write_csv(AREA_RESULTS_PATH, area_results)
    summary = build_summary(
        positive_rows, negative_rows, area_results, elapsed
    )
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")

    print_area_table(area_results)
    print("\n## Validation error-analysis results\n")
    print(summary)
    print(f"\nPositive details: {POSITIVE_DETAILS_PATH.resolve()}")
    print(f"Negative details: {NEGATIVE_DETAILS_PATH.resolve()}")
    print(f"Area results: {AREA_RESULTS_PATH.resolve()}")
    print(f"Summary: {SUMMARY_PATH.resolve()}")
    print("The final threshold and post-processing rule are not selected yet.")


if __name__ == "__main__":
    main()
