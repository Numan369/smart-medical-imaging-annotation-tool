import time
from pathlib import Path

import torch

from pneumothorax_dataloaders import create_dataloaders
from pneumothorax_model import PneumothoraxResNet34UNet


CHECKPOINT_PATH = (
    Path("checkpoints")
    / "balanced_decoder_stage_best.pth"
)
PREDICTION_THRESHOLD = 0.5
PROGRESS_INTERVAL = 50
EXPECTED_TRAINING_STAGE = "balanced_weighted_decoder_only"


def choose_device():
    """Use an NVIDIA GPU when available."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_checkpoint(path, model, device):
    """Load the model weights and return checkpoint metadata."""

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
        # Compatibility with older PyTorch versions.
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


def calculate_image_dice(predictions, targets):
    """Calculate one binary Dice score for each image."""

    dimensions = tuple(range(1, predictions.ndim))

    intersections = (
        predictions & targets
    ).sum(dim=dimensions).float()

    denominators = (
        predictions.sum(dim=dimensions).float()
        + targets.sum(dim=dimensions).float()
    )

    return torch.where(
        denominators > 0,
        (2.0 * intersections) / denominators,
        torch.ones_like(denominators),
    )


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate the best checkpoint on the untouched test set."""

    model.eval()

    total_dice = 0.0
    total_positive_dice = 0.0
    total_images = 0
    total_positive_images = 0
    total_negative_images = 0
    correctly_empty_negative_images = 0

    true_positive_pixels = 0
    false_positive_pixels = 0
    false_negative_pixels = 0

    start_time = time.perf_counter()
    total_batches = len(data_loader)

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

        logits = model(images)

        if logits.shape != targets.shape:
            raise ValueError(
                "Model output and target-mask shapes do not match."
            )

        predictions = (
            torch.sigmoid(logits)
            >= PREDICTION_THRESHOLD
        )

        dice_scores = calculate_image_dice(
            predictions,
            targets,
        )

        positive_cases = targets.flatten(start_dim=1).any(dim=1)
        negative_cases = ~positive_cases
        predicted_positive_cases = (
            predictions.flatten(start_dim=1).any(dim=1)
        )

        batch_size = images.shape[0]
        positive_count = int(positive_cases.sum().item())
        negative_count = int(negative_cases.sum().item())

        total_dice += dice_scores.sum().item()
        total_images += batch_size
        total_positive_images += positive_count
        total_negative_images += negative_count

        if positive_count > 0:
            total_positive_dice += (
                dice_scores[positive_cases].sum().item()
            )

        correctly_empty_negative_images += int(
            (
                negative_cases
                & ~predicted_positive_cases
            ).sum().item()
        )

        true_positive_pixels += int(
            (predictions & targets).sum().item()
        )
        false_positive_pixels += int(
            (predictions & ~targets).sum().item()
        )
        false_negative_pixels += int(
            (~predictions & targets).sum().item()
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
                f"  Test [{batch_number}/{total_batches}] "
                f"elapsed={elapsed_minutes:.1f} min"
            )

    if total_images == 0:
        raise ValueError("The test DataLoader contained no images.")

    if total_positive_images == 0:
        raise ValueError(
            "The test set contained no positive images."
        )

    if total_negative_images == 0:
        raise ValueError(
            "The test set contained no negative images."
        )

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

    pixel_precision = (
        true_positive_pixels / precision_denominator
        if precision_denominator > 0
        else 1.0
    )
    pixel_recall = (
        true_positive_pixels / recall_denominator
        if recall_denominator > 0
        else 1.0
    )
    pixel_iou = (
        true_positive_pixels / iou_denominator
        if iou_denominator > 0
        else 1.0
    )

    return {
        "images": total_images,
        "positive_images": total_positive_images,
        "negative_images": total_negative_images,
        "overall_dice": total_dice / total_images,
        "positive_case_dice": (
            total_positive_dice / total_positive_images
        ),
        "negative_empty_mask_accuracy": (
            correctly_empty_negative_images
            / total_negative_images
        ),
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_iou": pixel_iou,
        "seconds": time.perf_counter() - start_time,
    }


def print_results(results):
    print("\nFinal test results")
    print("------------------")
    print(f"Test images: {results['images']}")
    print(
        "Positive test images: "
        f"{results['positive_images']}"
    )
    print(
        "Negative test images: "
        f"{results['negative_images']}"
    )
    print(
        "Overall Dice (includes empty images): "
        f"{results['overall_dice']:.6f}"
    )
    print(
        "Positive-case Dice: "
        f"{results['positive_case_dice']:.6f}"
    )
    print(
        "Negative empty-mask accuracy: "
        f"{results['negative_empty_mask_accuracy']:.6f}"
    )
    print(
        "Foreground pixel precision: "
        f"{results['pixel_precision']:.6f}"
    )
    print(
        "Foreground pixel recall: "
        f"{results['pixel_recall']:.6f}"
    )
    print(
        "Foreground pixel IoU: "
        f"{results['pixel_iou']:.6f}"
    )
    print(
        "Evaluation time: "
        f"{results['seconds'] / 60.0:.1f} min"
    )


def main():
    device = choose_device()

    print("Best-checkpoint test evaluation")
    print("-------------------------------")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Prediction threshold: {PREDICTION_THRESHOLD}")
    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")

    data_loaders = create_dataloaders()
    test_loader = data_loaders["test"]

    print(f"Test batches: {len(test_loader)}")
    print("Loading model and best checkpoint...")

    # The checkpoint contains the complete encoder and decoder.
    # Avoid downloading ImageNet weights during evaluation.
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
    print(
        "Checkpoint best validation positive-case Dice: "
        f"{checkpoint.get('best_validation_positive_dice', float('nan')):.6f}"
    )
    print("\nEvaluating untouched test set...")

    results = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    print_results(results)


if __name__ == "__main__":
    main()
