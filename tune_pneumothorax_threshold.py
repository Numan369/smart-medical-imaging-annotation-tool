import time
from pathlib import Path

import torch

from pneumothorax_dataloaders import create_dataloaders
from pneumothorax_model import PneumothoraxResNet34UNet


CHECKPOINT_PATH = (
    Path("checkpoints")
    / "balanced_decoder_stage_best.pth"
)
THRESHOLDS = torch.tensor(
    [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    dtype=torch.float32,
)
PROGRESS_INTERVAL = 50
EXPECTED_TRAINING_STAGE = "balanced_weighted_decoder_only"


def choose_device():
    """Use an NVIDIA GPU when available."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_checkpoint(path, model, device):
    """Load the saved best model without changing it."""

    if not path.is_file():
        raise FileNotFoundError(
            "Best checkpoint was not found at: "
            f"{path.resolve()}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
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
            f"{training_stage!r}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint


@torch.no_grad()
def evaluate_thresholds(model, data_loader, device):
    """Measure candidate thresholds on the validation split."""

    model.eval()

    thresholds = THRESHOLDS.to(device)
    threshold_count = len(thresholds)

    positive_dice_sums = torch.zeros(
        threshold_count,
        dtype=torch.float64,
        device=device,
    )
    correctly_empty_negative_counts = torch.zeros(
        threshold_count,
        dtype=torch.long,
        device=device,
    )
    true_positive_pixels = torch.zeros_like(
        correctly_empty_negative_counts
    )
    false_positive_pixels = torch.zeros_like(
        correctly_empty_negative_counts
    )
    false_negative_pixels = torch.zeros_like(
        correctly_empty_negative_counts
    )

    positive_image_count = 0
    negative_image_count = 0
    total_batches = len(data_loader)
    start_time = time.perf_counter()

    for batch_number, batch in enumerate(
        data_loader,
        start=1,
    ):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        targets = batch["mask"].to(
            device,
            non_blocking=True,
        ) >= 0.5

        probabilities = torch.sigmoid(model(images))

        if probabilities.shape != targets.shape:
            raise ValueError(
                "Model output and target-mask shapes do not match."
            )

        # Shape: thresholds, batch, channel, height, width.
        predictions = probabilities.unsqueeze(0) >= thresholds.view(
            threshold_count,
            1,
            1,
            1,
            1,
        )
        expanded_targets = targets.unsqueeze(0)

        intersections = (
            predictions & expanded_targets
        ).sum(dim=(2, 3, 4)).float()
        denominators = (
            predictions.sum(dim=(2, 3, 4)).float()
            + expanded_targets.sum(dim=(2, 3, 4)).float()
        )
        dice_scores = torch.where(
            denominators > 0,
            (2.0 * intersections) / denominators,
            torch.ones_like(denominators),
        )

        positive_cases = targets.flatten(start_dim=1).any(dim=1)
        negative_cases = ~positive_cases
        predicted_positive_cases = predictions.flatten(
            start_dim=2
        ).any(dim=2)

        positive_count = int(positive_cases.sum().item())
        negative_count = int(negative_cases.sum().item())
        positive_image_count += positive_count
        negative_image_count += negative_count

        if positive_count > 0:
            positive_dice_sums += dice_scores[
                :, positive_cases
            ].sum(dim=1, dtype=torch.float64)

        correctly_empty_negative_counts += (
            ~predicted_positive_cases
            & negative_cases.unsqueeze(0)
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
                f"  Validation [{batch_number}/{total_batches}] "
                f"elapsed={elapsed_minutes:.1f} min"
            )

    if positive_image_count == 0 or negative_image_count == 0:
        raise ValueError(
            "Validation requires both positive and negative images."
        )

    positive_dice = (
        positive_dice_sums / positive_image_count
    )
    negative_empty_accuracy = (
        correctly_empty_negative_counts.double()
        / negative_image_count
    )

    precision_denominators = (
        true_positive_pixels + false_positive_pixels
    )
    recall_denominators = (
        true_positive_pixels + false_negative_pixels
    )

    pixel_precision = torch.where(
        precision_denominators > 0,
        true_positive_pixels.double()
        / precision_denominators.double(),
        torch.ones_like(precision_denominators, dtype=torch.float64),
    )
    pixel_recall = torch.where(
        recall_denominators > 0,
        true_positive_pixels.double()
        / recall_denominators.double(),
        torch.ones_like(recall_denominators, dtype=torch.float64),
    )

    # This harmonic mean prevents a threshold from winning by doing well
    # only on positive images or only on negative images.
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
        "negative_empty_accuracy": negative_empty_accuracy.cpu(),
        "pixel_precision": pixel_precision.cpu(),
        "pixel_recall": pixel_recall.cpu(),
        "balanced_score": balanced_score.cpu(),
        "positive_images": positive_image_count,
        "negative_images": negative_image_count,
        "seconds": time.perf_counter() - start_time,
    }


def print_results(results):
    """Print a compact comparison table and best balanced threshold."""

    print("\nValidation threshold results")
    print("----------------------------")
    print(
        "Threshold | Positive Dice | Negative empty | "
        "Pixel precision | Pixel recall | Balanced score"
    )

    for index, threshold in enumerate(results["thresholds"]):
        marker = "  < current" if abs(float(threshold) - 0.5) < 1e-6 else ""
        print(
            f"{float(threshold):9.2f} | "
            f"{float(results['positive_dice'][index]):13.6f} | "
            f"{float(results['negative_empty_accuracy'][index]):14.6f} | "
            f"{float(results['pixel_precision'][index]):15.6f} | "
            f"{float(results['pixel_recall'][index]):12.6f} | "
            f"{float(results['balanced_score'][index]):14.6f}"
            f"{marker}"
        )

    best_index = int(results["balanced_score"].argmax().item())
    best_threshold = float(results["thresholds"][best_index])

    print("\nSuggested balanced threshold")
    print("----------------------------")
    print(f"Threshold: {best_threshold:.2f}")
    print(
        "Positive-case Dice: "
        f"{float(results['positive_dice'][best_index]):.6f}"
    )
    print(
        "Negative empty-mask accuracy: "
        f"{float(results['negative_empty_accuracy'][best_index]):.6f}"
    )
    print(
        "Balanced score: "
        f"{float(results['balanced_score'][best_index]):.6f}"
    )
    print(
        "Evaluation time: "
        f"{results['seconds'] / 60.0:.1f} min"
    )
    print(
        "\nNo checkpoint or model parameters were changed."
    )


def main():
    device = choose_device()

    print("Validation threshold tuning")
    print("---------------------------")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print("Dataset split: validation (test set is not used)")
    print("Loading validation data...")

    data_loaders = create_dataloaders()
    validation_loader = data_loaders["validation"]

    print(f"Validation batches: {len(validation_loader)}")
    print("Loading model and best checkpoint...")

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=True,
    ).to(device)

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        model,
        device,
    )

    print(
        "Checkpoint epoch: "
        f"{checkpoint.get('completed_epoch', 'unknown')}"
    )
    print("\nChecking thresholds on validation set...")

    results = evaluate_thresholds(
        model=model,
        data_loader=validation_loader,
        device=device,
    )

    print_results(results)


if __name__ == "__main__":
    main()
