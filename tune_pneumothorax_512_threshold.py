import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pneumothorax_dataset import PneumothoraxDataset
from pneumothorax_model import PneumothoraxResNet34UNet


IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_WORKERS = 0
PROGRESS_INTERVAL = 50

CHECKPOINT_PATH = (
    Path("checkpoints") / "pneumothorax_512_best.pth"
)
OUTPUT_DIRECTORY = Path("validation_analysis") / "threshold_512"
RESULTS_PATH = OUTPUT_DIRECTORY / "threshold_results.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "threshold_summary.txt"

EXPECTED_TRAINING_STAGE = (
    "pneumothorax_512_augmented_progressive_finetune"
)
TRAINING_THRESHOLD = 0.35
THRESHOLDS = torch.arange(0.10, 0.701, 0.05)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_checkpoint(model, device):
    """Load the best upgraded checkpoint without modifying it."""

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "The upgraded checkpoint was not found: "
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

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain model_state_dict."
        )

    training_stage = checkpoint.get("training_stage")

    if training_stage != EXPECTED_TRAINING_STAGE:
        raise ValueError(
            "Unexpected checkpoint training stage: "
            f"{training_stage!r}. Expected: "
            f"{EXPECTED_TRAINING_STAGE!r}."
        )

    configuration = checkpoint.get("configuration", {})
    checkpoint_image_size = configuration.get("image_size")

    if checkpoint_image_size not in (None, IMAGE_SIZE):
        raise ValueError(
            "Checkpoint image size is not 512: "
            f"{checkpoint_image_size!r}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def create_validation_loader(device):
    """Create only the untouched validation loader."""

    validation_dataset = PneumothoraxDataset(
        split="validation",
        image_size=IMAGE_SIZE,
    )

    return DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


@torch.inference_mode()
def evaluate_thresholds(model, validation_loader, device):
    """Evaluate every candidate on the same validation predictions."""

    model.eval()
    thresholds = THRESHOLDS.to(device)
    threshold_count = thresholds.numel()

    positive_dice_sums = torch.zeros(
        threshold_count,
        dtype=torch.float64,
        device=device,
    )
    empty_positive_counts = torch.zeros(
        threshold_count,
        dtype=torch.long,
        device=device,
    )
    false_positive_image_counts = torch.zeros(
        threshold_count,
        dtype=torch.long,
        device=device,
    )
    true_positive_pixels = torch.zeros(
        threshold_count,
        dtype=torch.long,
        device=device,
    )
    false_positive_pixels = torch.zeros_like(true_positive_pixels)
    false_negative_pixels = torch.zeros_like(true_positive_pixels)

    positive_image_count = 0
    negative_image_count = 0
    total_batches = len(validation_loader)
    start_time = time.perf_counter()

    for batch_number, batch in enumerate(validation_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = (
            batch["mask"].to(device, non_blocking=True) >= 0.5
        )
        probabilities = torch.sigmoid(model(images))

        if probabilities.shape != targets.shape:
            raise ValueError(
                "Model output and target-mask shapes do not match."
            )

        predictions = probabilities.unsqueeze(0) >= thresholds.view(
            threshold_count,
            1,
            1,
            1,
            1,
        )
        expanded_targets = targets.unsqueeze(0)

        positive_cases = targets.flatten(start_dim=1).any(dim=1)
        negative_cases = ~positive_cases
        predicted_positive_cases = predictions.flatten(
            start_dim=2
        ).any(dim=2)

        positive_count = int(positive_cases.sum().item())
        negative_count = int(negative_cases.sum().item())
        positive_image_count += positive_count
        negative_image_count += negative_count

        intersections = (
            predictions & expanded_targets
        ).sum(dim=(2, 3, 4)).double()
        denominators = (
            predictions.sum(dim=(2, 3, 4)).double()
            + expanded_targets.sum(dim=(2, 3, 4)).double()
        )
        dice_scores = torch.where(
            denominators > 0,
            2.0 * intersections / denominators,
            torch.ones_like(denominators),
        )

        if positive_count:
            positive_dice_sums += dice_scores[
                :, positive_cases
            ].sum(dim=1)
            empty_positive_counts += (
                ~predicted_positive_cases[:, positive_cases]
            ).sum(dim=1)

        if negative_count:
            false_positive_image_counts += (
                predicted_positive_cases[:, negative_cases]
            ).sum(dim=1)

        true_positive_pixels += (
            predictions & expanded_targets
        ).sum(dim=(1, 2, 3, 4))
        false_positive_pixels += (
            predictions & ~expanded_targets
        ).sum(dim=(1, 2, 3, 4))
        false_negative_pixels += (
            ~predictions & expanded_targets
        ).sum(dim=(1, 2, 3, 4))

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

    if positive_image_count == 0 or negative_image_count == 0:
        raise ValueError(
            "Validation must contain positive and negative images."
        )

    positive_dice = positive_dice_sums / positive_image_count
    false_positive_image_rate = (
        false_positive_image_counts.double() / negative_image_count
    )
    negative_empty_accuracy = 1.0 - false_positive_image_rate

    precision_denominator = (
        true_positive_pixels + false_positive_pixels
    )
    recall_denominator = (
        true_positive_pixels + false_negative_pixels
    )
    iou_denominator = (
        true_positive_pixels
        + false_positive_pixels
        + false_negative_pixels
    )

    pixel_precision = torch.where(
        precision_denominator > 0,
        true_positive_pixels.double()
        / precision_denominator.double(),
        torch.ones_like(positive_dice),
    )
    pixel_recall = torch.where(
        recall_denominator > 0,
        true_positive_pixels.double() / recall_denominator.double(),
        torch.ones_like(positive_dice),
    )
    pixel_iou = torch.where(
        iou_denominator > 0,
        true_positive_pixels.double() / iou_denominator.double(),
        torch.ones_like(positive_dice),
    )

    balanced_score = torch.where(
        positive_dice + negative_empty_accuracy > 0,
        (
            2.0
            * positive_dice
            * negative_empty_accuracy
            / (positive_dice + negative_empty_accuracy)
        ),
        torch.zeros_like(positive_dice),
    )

    return {
        "thresholds": thresholds.cpu(),
        "positive_dice": positive_dice.cpu(),
        "empty_positive_counts": empty_positive_counts.cpu(),
        "false_positive_image_counts": (
            false_positive_image_counts.cpu()
        ),
        "false_positive_image_rate": false_positive_image_rate.cpu(),
        "negative_empty_accuracy": negative_empty_accuracy.cpu(),
        "pixel_precision": pixel_precision.cpu(),
        "pixel_recall": pixel_recall.cpu(),
        "pixel_iou": pixel_iou.cpu(),
        "balanced_score": balanced_score.cpu(),
        "positive_images": positive_image_count,
        "negative_images": negative_image_count,
        "elapsed_seconds": time.perf_counter() - start_time,
    }


def row_for_threshold(results, index):
    return {
        "threshold": float(results["thresholds"][index]),
        "positive_case_dice": float(
            results["positive_dice"][index]
        ),
        "empty_positive_predictions": int(
            results["empty_positive_counts"][index]
        ),
        "empty_positive_rate": (
            int(results["empty_positive_counts"][index])
            / results["positive_images"]
        ),
        "false_positive_negative_images": int(
            results["false_positive_image_counts"][index]
        ),
        "false_positive_negative_rate": float(
            results["false_positive_image_rate"][index]
        ),
        "negative_empty_accuracy": float(
            results["negative_empty_accuracy"][index]
        ),
        "pixel_precision": float(results["pixel_precision"][index]),
        "pixel_recall": float(results["pixel_recall"][index]),
        "pixel_iou": float(results["pixel_iou"][index]),
        "balanced_score": float(results["balanced_score"][index]),
    }


def save_results(results, checkpoint):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows = [
        row_for_threshold(results, index)
        for index in range(len(results["thresholds"]))
    ]

    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dice_best_index = max(
        range(len(rows)),
        key=lambda index: (
            rows[index]["positive_case_dice"],
            rows[index]["negative_empty_accuracy"],
            rows[index]["threshold"],
        ),
    )
    balanced_best_index = max(
        range(len(rows)),
        key=lambda index: (
            rows[index]["balanced_score"],
            rows[index]["positive_case_dice"],
            rows[index]["threshold"],
        ),
    )
    current_index = min(
        range(len(rows)),
        key=lambda index: abs(
            rows[index]["threshold"] - TRAINING_THRESHOLD
        ),
    )

    current = rows[current_index]
    dice_best = rows[dice_best_index]
    balanced_best = rows[balanced_best_index]

    summary_lines = [
        "Pneumothorax 512 x 512 validation threshold tuning",
        "===================================================",
        f"Checkpoint: {CHECKPOINT_PATH.resolve()}",
        "Dataset split: validation only",
        "Test split used: False",
        f"Validation images: "
        f"{results['positive_images'] + results['negative_images']}",
        f"Positive images: {results['positive_images']}",
        f"Negative images: {results['negative_images']}",
        f"Checkpoint epoch: "
        f"{checkpoint.get('completed_epoch', 'unknown')}",
        f"Checkpoint best positive Dice: "
        f"{checkpoint.get('best_validation_positive_dice', 'unknown')}",
        "",
        f"Training-time threshold ({TRAINING_THRESHOLD:.2f})",
        f"  Positive-case Dice: "
        f"{current['positive_case_dice']:.6f}",
        f"  Empty positive predictions: "
        f"{current['empty_positive_predictions']} / "
        f"{results['positive_images']} "
        f"({100.0 * current['empty_positive_rate']:.2f}%)",
        f"  False-positive negative images: "
        f"{current['false_positive_negative_images']} / "
        f"{results['negative_images']} "
        f"({100.0 * current['false_positive_negative_rate']:.2f}%)",
        "",
        "Highest positive-case Dice candidate",
        f"  Threshold: {dice_best['threshold']:.2f}",
        f"  Positive-case Dice: "
        f"{dice_best['positive_case_dice']:.6f}",
        f"  Empty positive predictions: "
        f"{dice_best['empty_positive_predictions']} / "
        f"{results['positive_images']} "
        f"({100.0 * dice_best['empty_positive_rate']:.2f}%)",
        f"  False-positive negative images: "
        f"{dice_best['false_positive_negative_images']} / "
        f"{results['negative_images']} "
        f"({100.0 * dice_best['false_positive_negative_rate']:.2f}%)",
        "",
        "Highest balanced-score candidate",
        f"  Threshold: {balanced_best['threshold']:.2f}",
        f"  Positive-case Dice: "
        f"{balanced_best['positive_case_dice']:.6f}",
        f"  Negative empty-mask accuracy: "
        f"{balanced_best['negative_empty_accuracy']:.6f}",
        f"  Balanced score: {balanced_best['balanced_score']:.6f}",
        "",
        f"Evaluation time: "
        f"{results['elapsed_seconds'] / 60.0:.1f} min",
        "No checkpoint or model parameters were changed.",
    ]

    summary = "\n".join(summary_lines)
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")
    return rows, current_index, dice_best_index, balanced_best_index, summary


def print_table(rows, current_index, dice_best_index, balanced_best_index):
    print("\nValidation threshold results")
    print("----------------------------")
    print(
        "Thr | Pos Dice | Empty pos | FP negatives | "
        "Px precision | Px recall | Px IoU | Balanced"
    )

    for index, row in enumerate(rows):
        markers = []
        if index == current_index:
            markers.append("training")
        if index == dice_best_index:
            markers.append("best Dice")
        if index == balanced_best_index:
            markers.append("best balanced")
        marker_text = f"  < {', '.join(markers)}" if markers else ""

        print(
            f"{row['threshold']:.2f} | "
            f"{row['positive_case_dice']:.6f} | "
            f"{row['empty_positive_predictions']:3d}"
            f"/{row['empty_positive_rate'] * 100:5.2f}% | "
            f"{row['false_positive_negative_images']:3d}"
            f"/{row['false_positive_negative_rate'] * 100:5.2f}% | "
            f"{row['pixel_precision']:.6f} | "
            f"{row['pixel_recall']:.6f} | "
            f"{row['pixel_iou']:.6f} | "
            f"{row['balanced_score']:.6f}"
            f"{marker_text}"
        )


def main():
    device = choose_device()

    print("Pneumothorax 512 x 512 threshold tuning")
    print("----------------------------------------")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print("Dataset split: validation only")
    print("Test split: not created or accessed")
    print("Training: disabled")

    validation_loader = create_validation_loader(device)
    print(f"Validation images: {len(validation_loader.dataset):,}")
    print(f"Validation batches: {len(validation_loader):,}")

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=True,
    ).to(device)
    checkpoint = load_checkpoint(model, device)

    print(
        "Checkpoint epoch: "
        f"{checkpoint.get('completed_epoch', 'unknown')}"
    )
    print("\nEvaluating candidate thresholds...")

    results = evaluate_thresholds(
        model,
        validation_loader,
        device,
    )
    (
        rows,
        current_index,
        dice_best_index,
        balanced_best_index,
        summary,
    ) = save_results(results, checkpoint)

    print_table(
        rows,
        current_index,
        dice_best_index,
        balanced_best_index,
    )
    print("\n" + summary)
    print(f"\nDetailed CSV: {RESULTS_PATH.resolve()}")
    print(f"Summary file: {SUMMARY_PATH.resolve()}")
    print("The final threshold has not been selected yet.")


if __name__ == "__main__":
    main()
