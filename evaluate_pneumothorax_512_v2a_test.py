"""One-time evaluation of the selected negative-aware model on the test set.

This script never trains, tunes the threshold, or changes the checkpoint.
It loads the fixed threshold saved in the best validation-selected checkpoint
and evaluates every image in the previously untouched test split exactly once.
"""

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pneumothorax_dataset import PneumothoraxDataset
from pneumothorax_model import PneumothoraxResNet34UNet


CHECKPOINT_PATH = Path(
    "/content/drive/MyDrive/SmartMedicalImagingColab/checkpoints/"
    "pneumothorax_512_v2a_hard_negative_best.pth"
)
RESULTS_PATH = Path(
    "/content/drive/MyDrive/SmartMedicalImagingColab/"
    "pneumothorax_512_v2a_hard_negative_test_results.json"
)
EXPECTED_TRAINING_STAGE = "pneumothorax_512_v2a_hard_negative_finetune"
EXPECTED_COMPLETED_EPOCH = 4
EXPECTED_IMAGE_SIZE = 512
EXPECTED_PREDICTION_THRESHOLD = 0.35
EXPECTED_TEST_IMAGES = 1205
EXPECTED_POSITIVE_IMAGES = 267
EXPECTED_NEGATIVE_IMAGES = 938
BATCH_SIZE = 2
PROGRESS_INTERVAL = 100


def load_torch_checkpoint(path, device):
    """Load a checkpoint on both older and newer PyTorch versions."""

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def harmonic_mean(first, second):
    if first <= 0.0 or second <= 0.0:
        return 0.0
    return 2.0 * first * second / (first + second)


def validate_checkpoint(checkpoint):
    """Reject the wrong checkpoint or a changed evaluation configuration."""

    if checkpoint.get("training_stage") != EXPECTED_TRAINING_STAGE:
        raise ValueError(
            "Unexpected training stage: "
            f"{checkpoint.get('training_stage')!r}"
        )

    if checkpoint.get("completed_epoch") != EXPECTED_COMPLETED_EPOCH:
        raise ValueError(
            "Expected the validation-selected epoch-4 checkpoint, found "
            f"epoch {checkpoint.get('completed_epoch')!r}."
        )

    configuration = checkpoint.get("configuration", {})
    if configuration.get("test_split_used") is not False:
        raise ValueError(
            "Checkpoint metadata does not confirm an untouched test split."
        )

    image_size = int(configuration.get("image_size", -1))
    threshold = float(configuration.get("prediction_threshold", -1.0))

    if image_size != EXPECTED_IMAGE_SIZE:
        raise ValueError(f"Unexpected image size: {image_size}")

    if not math.isclose(
        threshold,
        EXPECTED_PREDICTION_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Unexpected prediction threshold: {threshold}")

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")

    return image_size, threshold


@torch.inference_mode()
def evaluate(model, data_loader, device, threshold, image_size):
    """Calculate segmentation and healthy-image metrics on the full test set."""

    model.eval()

    samples = 0
    positive_samples = 0
    negative_samples = 0
    dice_sum = 0.0
    positive_dice_sum = 0.0
    empty_positive_predictions = 0
    empty_negative_predictions = 0
    negative_predicted_fraction_sum = 0.0
    true_positive_pixels = 0
    false_positive_pixels = 0
    false_negative_pixels = 0

    started = time.perf_counter()
    total_batches = len(data_loader)
    pixels_per_image = float(image_size * image_size)
    use_amp = device.type == "cuda"

    for batch_number, batch in enumerate(data_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True) >= 0.5

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)

        if logits.shape != targets.shape:
            raise ValueError(
                "Model output and target-mask shapes do not match: "
                f"{tuple(logits.shape)} versus {tuple(targets.shape)}"
            )

        predictions = torch.sigmoid(logits) >= threshold
        dimensions = tuple(range(1, targets.ndim))
        intersections = (predictions & targets).sum(dim=dimensions).float()
        predicted_areas = predictions.sum(dim=dimensions).float()
        target_areas = targets.sum(dim=dimensions).float()
        denominators = predicted_areas + target_areas
        dice = torch.where(
            denominators > 0,
            2.0 * intersections / denominators,
            torch.ones_like(denominators),
        )

        positive_cases = target_areas > 0
        negative_cases = ~positive_cases
        batch_size = images.shape[0]

        samples += batch_size
        dice_sum += dice.sum().item()

        if positive_cases.any():
            count = int(positive_cases.sum().item())
            positive_samples += count
            positive_dice_sum += dice[positive_cases].sum().item()
            empty_positive_predictions += int(
                (predicted_areas[positive_cases] == 0).sum().item()
            )

        if negative_cases.any():
            count = int(negative_cases.sum().item())
            negative_samples += count
            empty_negative_predictions += int(
                (predicted_areas[negative_cases] == 0).sum().item()
            )
            negative_predicted_fraction_sum += (
                predicted_areas[negative_cases].sum().item()
                / pixels_per_image
            )

        true_positive_pixels += int((predictions & targets).sum().item())
        false_positive_pixels += int((predictions & ~targets).sum().item())
        false_negative_pixels += int((~predictions & targets).sum().item())

        if (
            batch_number == 1
            or batch_number % PROGRESS_INTERVAL == 0
            or batch_number == total_batches
        ):
            elapsed_minutes = (time.perf_counter() - started) / 60.0
            print(
                f"Test [{batch_number}/{total_batches}] "
                f"elapsed={elapsed_minutes:.1f} min",
                flush=True,
            )

    if samples != EXPECTED_TEST_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_IMAGES} test images, found {samples}."
        )
    if positive_samples != EXPECTED_POSITIVE_IMAGES:
        raise ValueError(
            "Expected "
            f"{EXPECTED_POSITIVE_IMAGES} positive test images, "
            f"found {positive_samples}."
        )
    if negative_samples != EXPECTED_NEGATIVE_IMAGES:
        raise ValueError(
            "Expected "
            f"{EXPECTED_NEGATIVE_IMAGES} negative test images, "
            f"found {negative_samples}."
        )

    positive_dice = positive_dice_sum / positive_samples
    negative_empty_accuracy = (
        empty_negative_predictions / negative_samples
    )
    precision_denominator = true_positive_pixels + false_positive_pixels
    recall_denominator = true_positive_pixels + false_negative_pixels
    iou_denominator = (
        true_positive_pixels + false_positive_pixels + false_negative_pixels
    )

    return {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": EXPECTED_COMPLETED_EPOCH,
        "image_size": image_size,
        "prediction_threshold": threshold,
        "test_images": samples,
        "positive_test_images": positive_samples,
        "negative_test_images": negative_samples,
        "overall_dice_including_empty_images": dice_sum / samples,
        "positive_case_dice": positive_dice,
        "negative_empty_mask_accuracy": negative_empty_accuracy,
        "joint_selection_score": harmonic_mean(
            positive_dice,
            negative_empty_accuracy,
        ),
        "empty_positive_predictions": empty_positive_predictions,
        "positive_miss_rate": empty_positive_predictions / positive_samples,
        "false_positive_negative_images": (
            negative_samples - empty_negative_predictions
        ),
        "false_positive_negative_rate": (
            1.0 - negative_empty_accuracy
        ),
        "mean_predicted_area_on_negatives": (
            negative_predicted_fraction_sum / negative_samples
        ),
        "foreground_pixel_precision": (
            true_positive_pixels / precision_denominator
            if precision_denominator > 0
            else 1.0
        ),
        "foreground_pixel_recall": (
            true_positive_pixels / recall_denominator
            if recall_denominator > 0
            else 1.0
        ),
        "foreground_pixel_iou": (
            true_positive_pixels / iou_denominator
            if iou_denominator > 0
            else 1.0
        ),
        "evaluation_seconds": time.perf_counter() - started,
        "test_split_used_for_training_or_tuning": False,
    }


def print_results(results):
    print("\nFINAL UNTOUCHED TEST RESULTS")
    print("----------------------------")
    print(f"Test images: {results['test_images']}")
    print(f"Positive test images: {results['positive_test_images']}")
    print(f"Negative test images: {results['negative_test_images']}")
    print(
        "Overall Dice (includes empty negatives): "
        f"{results['overall_dice_including_empty_images']:.6f}"
    )
    print(f"Positive-case Dice: {results['positive_case_dice']:.6f}")
    print(
        "Negative empty-mask accuracy: "
        f"{results['negative_empty_mask_accuracy']:.6f}"
    )
    print(f"Joint score: {results['joint_selection_score']:.6f}")
    print(
        "Empty positive predictions: "
        f"{results['empty_positive_predictions']} / "
        f"{results['positive_test_images']} "
        f"({100.0 * results['positive_miss_rate']:.2f}%)"
    )
    print(
        "False-positive negative images: "
        f"{results['false_positive_negative_images']} / "
        f"{results['negative_test_images']} "
        f"({100.0 * results['false_positive_negative_rate']:.2f}%)"
    )
    print(
        "Mean predicted area on negatives: "
        f"{100.0 * results['mean_predicted_area_on_negatives']:.4f}%"
    )
    print(
        "Foreground pixel precision: "
        f"{results['foreground_pixel_precision']:.6f}"
    )
    print(
        "Foreground pixel recall: "
        f"{results['foreground_pixel_recall']:.6f}"
    )
    print(
        "Foreground pixel IoU: "
        f"{results['foreground_pixel_iou']:.6f}"
    )
    print(
        "Evaluation time: "
        f"{results['evaluation_seconds'] / 60.0:.1f} min"
    )


def main():
    device = choose_device()

    print("Negative-aware model: one-time final test evaluation")
    print("----------------------------------------------------")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Best checkpoint was not found: {CHECKPOINT_PATH}"
        )

    checkpoint = load_torch_checkpoint(CHECKPOINT_PATH, device)
    image_size, threshold = validate_checkpoint(checkpoint)

    print(f"Checkpoint epoch: {checkpoint['completed_epoch']}")
    print(f"Image size: {image_size} x {image_size}")
    print(f"Fixed prediction threshold: {threshold}")
    print("Loading model without downloading ImageNet weights...")

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # The test split is instantiated only after the selected checkpoint and
    # its fixed validation-derived configuration have passed every check.
    test_dataset = PneumothoraxDataset(
        split="test",
        image_size=image_size,
    )
    number_of_workers = 2 if device.type == "cuda" else 0
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=number_of_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    print(f"Test batches: {len(test_loader)}")
    print("Starting the one-time full test evaluation...\n")

    results = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        threshold=threshold,
        image_size=image_size,
    )
    print_results(results)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nResults saved to: {RESULTS_PATH}")
    print("The test threshold was not tuned or changed.")


if __name__ == "__main__":
    main()
