"""Evaluate one frozen pneumothorax checkpoint on the full test split.

This script applies the frozen V4A operating configuration to either V4A
or V4B so their predictions are compared under identical conditions:

* probability threshold: 0.15
* minimum connected component: 112 pixels at 512 x 512
* 8-connectivity

Run this only after checkpoint selection and operating-configuration freezing.
Do not use these test results to change checkpoints, thresholds, or filtering.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from pneumothorax_dataset import PneumothoraxDataset
from pneumothorax_model import PneumothoraxResNet34UNet


IMAGE_SIZE = 512
DEFAULT_THRESHOLD = 0.15
DEFAULT_MINIMUM_COMPONENT_PIXELS = 112
BATCH_SIZE = 2


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen V4 checkpoint on the complete test split "
            "using a fixed threshold and component-size filter."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )
    parser.add_argument(
        "--minimum-component-pixels",
        type=int,
        default=DEFAULT_MINIMUM_COMPONENT_PIXELS,
    )
    return parser.parse_args()


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def remove_small_components(binary_mask, minimum_pixels):
    """Remove 8-connected foreground regions smaller than minimum_pixels."""

    binary_mask = np.asarray(binary_mask, dtype=bool)
    if not binary_mask.any():
        return binary_mask

    structure = np.ones((3, 3), dtype=np.uint8)
    labelled_mask, component_count = ndimage.label(
        binary_mask,
        structure=structure,
    )

    if component_count == 0:
        return np.zeros_like(binary_mask, dtype=bool)

    component_sizes = np.bincount(labelled_mask.ravel())
    retained_labels = component_sizes >= minimum_pixels
    retained_labels[0] = False
    return retained_labels[labelled_mask]


def empty_accumulator():
    return {
        "images": 0,
        "positive_images": 0,
        "negative_images": 0,
        "positive_dice_sum": 0.0,
        "positive_iou_sum": 0.0,
        "overall_case_dice_sum": 0.0,
        "detected_positive_images": 0,
        "empty_negative_predictions": 0,
        "false_positive_negative_images": 0,
        "missed_positive_images": 0,
        "negative_predicted_fraction_sum": 0.0,
        "predicted_positive_images": 0,
    }


def update_accumulator(accumulator, prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    target_area = int(target.sum())
    prediction_area = int(prediction.sum())
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())

    accumulator["images"] += 1
    if prediction_area > 0:
        accumulator["predicted_positive_images"] += 1

    if target_area > 0:
        accumulator["positive_images"] += 1
        dice = (
            2.0 * intersection / (prediction_area + target_area)
            if prediction_area + target_area > 0
            else 0.0
        )
        iou = intersection / union if union > 0 else 0.0
        accumulator["positive_dice_sum"] += dice
        accumulator["positive_iou_sum"] += iou
        accumulator["overall_case_dice_sum"] += dice

        if prediction_area > 0:
            accumulator["detected_positive_images"] += 1
        else:
            accumulator["missed_positive_images"] += 1
    else:
        accumulator["negative_images"] += 1
        accumulator["negative_predicted_fraction_sum"] += (
            prediction_area / float(IMAGE_SIZE * IMAGE_SIZE)
        )

        if prediction_area == 0:
            accumulator["empty_negative_predictions"] += 1
            accumulator["overall_case_dice_sum"] += 1.0
        else:
            accumulator["false_positive_negative_images"] += 1


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def finalize_results(accumulator, elapsed_seconds, arguments):
    positive_count = accumulator["positive_images"]
    negative_count = accumulator["negative_images"]
    sensitivity = safe_divide(
        accumulator["detected_positive_images"],
        positive_count,
    )
    specificity = safe_divide(
        accumulator["empty_negative_predictions"],
        negative_count,
    )

    return {
        "name": arguments.name,
        "checkpoint": str(arguments.checkpoint),
        "split": "test",
        "testSplitUsed": True,
        "imageSize": [IMAGE_SIZE, IMAGE_SIZE],
        "threshold": arguments.threshold,
        "minimumComponentPixels": (
            arguments.minimum_component_pixels
        ),
        "connectivity": 8,
        "images": accumulator["images"],
        "positiveImages": positive_count,
        "negativeImages": negative_count,
        "positiveDice": safe_divide(
            accumulator["positive_dice_sum"], positive_count
        ),
        "positiveIoU": safe_divide(
            accumulator["positive_iou_sum"], positive_count
        ),
        "overallCaseWiseDice": safe_divide(
            accumulator["overall_case_dice_sum"],
            accumulator["images"],
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balancedDetectionAccuracy": (sensitivity + specificity) / 2.0,
        "missedPositiveImages": accumulator["missed_positive_images"],
        "falsePositiveNegativeImages": accumulator[
            "false_positive_negative_images"
        ],
        "negativeMeanPredictedFraction": safe_divide(
            accumulator["negative_predicted_fraction_sum"],
            negative_count,
        ),
        "predictedPositiveImages": accumulator[
            "predicted_positive_images"
        ],
        "elapsedSeconds": elapsed_seconds,
    }


def print_results(results):
    print("\nFrozen common-condition test results")
    print("------------------------------------")
    print(f"Model: {results['name']}")
    print(f"Images: {results['images']:,}")
    print(f"Positive images: {results['positiveImages']:,}")
    print(f"Negative images: {results['negativeImages']:,}")
    print(f"Threshold: {results['threshold']:.4f}")
    print(
        "Minimum component pixels: "
        f"{results['minimumComponentPixels']}"
    )
    print(f"Positive Dice: {results['positiveDice']:.6f}")
    print(f"Positive IoU: {results['positiveIoU']:.6f}")
    print(
        "Overall case-wise Dice: "
        f"{results['overallCaseWiseDice']:.6f}"
    )
    print(f"Sensitivity: {results['sensitivity']:.6f}")
    print(f"Specificity: {results['specificity']:.6f}")
    print(
        "Balanced detection accuracy: "
        f"{results['balancedDetectionAccuracy']:.6f}"
    )
    print(
        "Missed positive images: "
        f"{results['missedPositiveImages']} / "
        f"{results['positiveImages']}"
    )
    print(
        "False-positive negative images: "
        f"{results['falsePositiveNegativeImages']} / "
        f"{results['negativeImages']}"
    )
    print(
        "Mean predicted area on negatives: "
        f"{100.0 * results['negativeMeanPredictedFraction']:.4f}%"
    )
    print(f"Elapsed: {results['elapsedSeconds'] / 60.0:.2f} min")
    print("Test split used: True")


def main():
    arguments = parse_arguments()
    if not 0.0 <= arguments.threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if arguments.minimum_component_pixels < 1:
        raise ValueError("minimum-component-pixels must be at least 1")

    device = choose_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {arguments.checkpoint}")
    print("Loading complete test split for one frozen evaluation...")

    dataset = PneumothoraxDataset(
        split="test",
        image_size=IMAGE_SIZE,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=True,
    ).to(device)
    checkpoint = load_checkpoint(arguments.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    accumulator = empty_accumulator()
    start_time = time.perf_counter()
    total_batches = len(loader)

    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].cpu().numpy() >= 0.5

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)

            probabilities = torch.sigmoid(logits).float().cpu().numpy()

            for index in range(images.shape[0]):
                prediction = probabilities[index, 0] >= arguments.threshold
                prediction = remove_small_components(
                    prediction,
                    arguments.minimum_component_pixels,
                )
                update_accumulator(
                    accumulator,
                    prediction,
                    targets[index, 0],
                )

            if (
                batch_number == 1
                or batch_number % 100 == 0
                or batch_number == total_batches
            ):
                elapsed_minutes = (
                    time.perf_counter() - start_time
                ) / 60.0
                print(
                    f"Test [{batch_number}/{total_batches}] "
                    f"elapsed={elapsed_minutes:.1f} min"
                )

    results = finalize_results(
        accumulator,
        time.perf_counter() - start_time,
        arguments,
    )
    print_results(results)

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    output_path = (
        arguments.output_directory
        / f"{arguments.name}_common_test.json"
    )
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=4)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()