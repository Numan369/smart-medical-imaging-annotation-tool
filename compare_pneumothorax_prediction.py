"""Compare the locked V1 pneumothorax prediction with a test ground-truth mask.

Run this script from the project root. It uses only the labelled internal test
split and never trains or changes the checkpoint.
"""

from argparse import ArgumentParser
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from pneumothorax_dataset import PneumothoraxDataset
from pneumothorax_model import PneumothoraxResNet34UNet


DEFAULT_CHECKPOINT = (
    Path("checkpoints") / "pneumothorax_512_negative_aware_best.pth"
)
EXPECTED_STAGE = "pneumothorax_512_negative_aware_finetune"
EXPECTED_EPOCH = 4
EXPECTED_IMAGE_SIZE = 512
EXPECTED_THRESHOLD = 0.35


def parse_arguments():
    parser = ArgumentParser(
        description=(
            "Show a labelled test X-ray, its actual mask, and the locked "
            "V1 model prediction."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--image-id",
        help="Exact ImageId from the internal test split.",
    )
    selection.add_argument(
        "--index",
        type=int,
        help="Zero-based index within the complete internal test split.",
    )
    selection.add_argument(
        "--random",
        action="store_true",
        help="Select a random image from the complete internal test split.",
    )
    selection.add_argument(
        "--random-positive",
        action="store_true",
        help="Select a random positive image with an actual pneumothorax mask.",
    )
    selection.add_argument(
        "--random-negative",
        action="store_true",
        help="Select a random negative image whose actual mask is empty.",
    )
    parser.add_argument(
        "--include-negative",
        action="store_true",
        help=(
            "When no ID/index is supplied, select the first test image even "
            "if it is negative. The default selects the first positive case."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Locked V1 best-checkpoint path.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional PNG output path. The comparison is always displayed.",
    )
    return parser.parse_args()


def load_checkpoint(path, device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {path.resolve()}")

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if checkpoint.get("training_stage") != EXPECTED_STAGE:
        raise ValueError(
            "Wrong checkpoint stage: "
            f"{checkpoint.get('training_stage')!r}; expected {EXPECTED_STAGE!r}."
        )

    if checkpoint.get("completed_epoch") != EXPECTED_EPOCH:
        raise ValueError(
            "Wrong checkpoint epoch: "
            f"{checkpoint.get('completed_epoch')!r}; expected {EXPECTED_EPOCH}."
        )

    configuration = checkpoint.get("configuration", {})
    image_size = int(configuration.get("image_size", -1))
    threshold = float(configuration.get("prediction_threshold", -1.0))

    if image_size != EXPECTED_IMAGE_SIZE:
        raise ValueError(
            f"Wrong image size: {image_size}; expected {EXPECTED_IMAGE_SIZE}."
        )

    if not np.isclose(threshold, EXPECTED_THRESHOLD):
        raise ValueError(
            f"Wrong threshold: {threshold}; expected {EXPECTED_THRESHOLD}."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")

    return checkpoint, image_size, threshold


def select_index(
    dataset,
    image_id,
    index,
    include_negative,
    random_image,
    random_positive,
    random_negative,
):
    if image_id is not None:
        for candidate_index, row in enumerate(dataset.rows):
            if row["ImageId"] == image_id:
                return candidate_index
        raise ValueError(f"ImageId is not present in the test split: {image_id}")

    if index is not None:
        if index < 0 or index >= len(dataset):
            raise IndexError(
                f"Test index must be between 0 and {len(dataset) - 1}."
            )
        return index

    if random_image:
        return random.randrange(len(dataset))

    if random_positive:
        candidates = [
            candidate_index
            for candidate_index, row in enumerate(dataset.rows)
            if int(row["HasPneumothorax"]) == 1
        ]
        return random.choice(candidates)

    if random_negative:
        candidates = [
            candidate_index
            for candidate_index, row in enumerate(dataset.rows)
            if int(row["HasPneumothorax"]) == 0
        ]
        return random.choice(candidates)

    if include_negative:
        return 0

    return next(
        candidate_index
        for candidate_index, row in enumerate(dataset.rows)
        if int(row["HasPneumothorax"]) == 1
    )


def calculate_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    true_positive = int(np.logical_and(prediction, target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())
    predicted_pixels = int(prediction.sum())
    actual_pixels = int(target.sum())
    union = int(np.logical_or(prediction, target).sum())

    denominator = predicted_pixels + actual_pixels
    if denominator == 0:
        dice = 1.0
    else:
        dice = 2.0 * true_positive / denominator

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else (1.0 if actual_pixels == 0 else 0.0)
    )
    recall = (
        true_positive / recall_denominator
        if recall_denominator
        else (1.0 if predicted_pixels == 0 else 0.0)
    )
    iou = true_positive / union if union else 1.0

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "actual_pixels": actual_pixels,
        "predicted_pixels": predicted_pixels,
    }


def coloured_overlay(image, mask, colour):
    rgb = np.repeat(image[..., None], 3, axis=2)
    colour_array = np.asarray(colour, dtype=np.float32)
    alpha = 0.45
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * colour_array
    return np.clip(rgb, 0.0, 1.0)


def comparison_overlay(image, actual, predicted):
    rgb = np.repeat(image[..., None], 3, axis=2)
    true_positive = np.logical_and(actual, predicted)
    missed = np.logical_and(actual, ~predicted)
    extra = np.logical_and(~actual, predicted)

    # Green: overlap, red: missed ground truth, blue: extra prediction.
    rgb[true_positive] = 0.45 * rgb[true_positive] + 0.55 * np.array(
        [0.0, 1.0, 0.0]
    )
    rgb[missed] = 0.45 * rgb[missed] + 0.55 * np.array([1.0, 0.0, 0.0])
    rgb[extra] = 0.45 * rgb[extra] + 0.55 * np.array([0.0, 0.4, 1.0])
    return np.clip(rgb, 0.0, 1.0)


def main():
    arguments = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint, image_size, threshold = load_checkpoint(
        arguments.checkpoint,
        device,
    )

    print("Creating the labelled internal test dataset...")
    dataset = PneumothoraxDataset(split="test", image_size=image_size)
    selected_index = select_index(
        dataset,
        arguments.image_id,
        arguments.index,
        arguments.include_negative,
        arguments.random,
        arguments.random_positive,
        arguments.random_negative,
    )
    sample = dataset[selected_index]

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_tensor = sample["image"]
    with torch.inference_mode():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probabilities = torch.sigmoid(logits)
        prediction_tensor = probabilities >= threshold

    image = image_tensor.squeeze(0).cpu().numpy()
    actual = sample["mask"].squeeze(0).cpu().numpy() >= 0.5
    predicted = prediction_tensor.squeeze(0).squeeze(0).cpu().numpy()
    metrics = calculate_metrics(predicted, actual)

    label = int(sample["label"].item())
    print("\nComparison details")
    print("Split: test")
    print("Test index:", selected_index)
    print("ImageId:", sample["image_id"])
    print("Actual image-level label:", label)
    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["completed_epoch"])
    print("Image size:", image_size)
    print("Fixed threshold:", threshold)
    print("Actual mask pixels:", metrics["actual_pixels"])
    print("Predicted mask pixels:", metrics["predicted_pixels"])
    print(f"Dice: {metrics['dice']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"IoU: {metrics['iou']:.4f}")

    figure, axes = plt.subplots(1, 4, figsize=(20, 6))
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original test X-ray")
    axes[1].imshow(coloured_overlay(image, actual, [1.0, 0.85, 0.0]))
    axes[1].set_title(f"Actual mask\n{metrics['actual_pixels']:,} pixels")
    axes[2].imshow(coloured_overlay(image, predicted, [0.0, 0.8, 1.0]))
    axes[2].set_title(
        f"V1 prediction (threshold {threshold:.2f})\n"
        f"{metrics['predicted_pixels']:,} pixels"
    )
    axes[3].imshow(comparison_overlay(image, actual, predicted))
    axes[3].set_title(
        "Comparison\nGreen=overlap, Red=missed, Blue=extra"
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"ImageId: {sample['image_id']} | Dice: {metrics['dice']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | IoU: {metrics['iou']:.4f}",
        fontsize=11,
    )
    figure.tight_layout()

    if arguments.save is not None:
        arguments.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(arguments.save, dpi=160, bbox_inches="tight")
        print("Saved comparison:", arguments.save.resolve())

    plt.show()


if __name__ == "__main__":
    main()
