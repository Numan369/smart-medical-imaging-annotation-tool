import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from pneumothorax_dataset import PneumothoraxDataset
from pneumothorax_model import PneumothoraxResNet34UNet


IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_WORKERS = 0
PREDICTION_THRESHOLD = 0.35
EXAMPLES_PER_GROUP = 6
PROGRESS_INTERVAL = 50

CHECKPOINT_PATH = Path("checkpoints") / "pneumothorax_512_best.pth"
EXPECTED_TRAINING_STAGE = "pneumothorax_512_augmented_progressive_finetune"

ERROR_DIRECTORY = Path("validation_analysis") / "errors_512"
POSITIVE_DETAILS_PATH = ERROR_DIRECTORY / "positive_case_details.csv"
NEGATIVE_DETAILS_PATH = ERROR_DIRECTORY / "negative_case_details.csv"
OUTPUT_DIRECTORY = ERROR_DIRECTORY / "visual_audit"
HARDEST_POSITIVE_PATH = OUTPUT_DIRECTORY / "hardest_positive_localizations.png"
LARGEST_FALSE_POSITIVE_PATH = OUTPUT_DIRECTORY / "largest_false_positives.png"
SELECTION_PATH = OUTPUT_DIRECTORY / "visual_audit_selection.txt"


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
            f"Checkpoint image size is {checkpoint_size!r}, not {IMAGE_SIZE}."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required analysis CSV was not found: {path.resolve()}\n"
            "Run analyze_pneumothorax_512_errors.py first."
        )

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def choose_cases():
    positive_rows = read_rows(POSITIVE_DETAILS_PATH)
    negative_rows = read_rows(NEGATIVE_DETAILS_PATH)

    hardest_positive_rows = sorted(
        positive_rows,
        key=lambda row: (
            float(row["dice"]),
            float(row["target_area_percent"]),
            -float(row["predicted_pixels"]),
        ),
    )[:EXAMPLES_PER_GROUP]

    false_positive_rows = [
        row for row in negative_rows if int(row["false_positive"]) == 1
    ]
    largest_false_positive_rows = sorted(
        false_positive_rows,
        key=lambda row: (
            float(row["predicted_pixels"]),
            float(row["maximum_probability"]),
        ),
        reverse=True,
    )[:EXAMPLES_PER_GROUP]

    if len(hardest_positive_rows) < EXAMPLES_PER_GROUP:
        raise ValueError("Not enough positive cases were found in the CSV.")
    if len(largest_false_positive_rows) < EXAMPLES_PER_GROUP:
        raise ValueError("Not enough false-positive cases were found in the CSV.")

    return hardest_positive_rows, largest_false_positive_rows


def convert_sample(image_id, image, target, probability):
    image_array = image.squeeze().detach().cpu().numpy()
    target_array = target.squeeze().detach().cpu().numpy() >= 0.5
    probability_array = probability.squeeze().detach().cpu().numpy()
    prediction_array = probability_array >= PREDICTION_THRESHOLD

    target_pixels = int(target_array.sum())
    predicted_pixels = int(prediction_array.sum())
    intersection = int(np.logical_and(target_array, prediction_array).sum())
    denominator = target_pixels + predicted_pixels
    dice = 2.0 * intersection / denominator if denominator else 1.0

    return {
        "image_id": str(image_id),
        "image": image_array,
        "target": target_array,
        "probability": probability_array,
        "prediction": prediction_array,
        "target_pixels": target_pixels,
        "predicted_pixels": predicted_pixels,
        "maximum_probability": float(probability_array.max()),
        "dice": dice,
    }


@torch.inference_mode()
def collect_selected_samples(model, loader, device, selected_ids):
    model.eval()
    collected = {}
    total_batches = len(loader)

    for batch_number, batch in enumerate(loader, start=1):
        matching_indices = [
            index
            for index, image_id in enumerate(batch["image_id"])
            if str(image_id) in selected_ids
        ]

        if matching_indices:
            images = batch["image"][matching_indices].to(
                device, non_blocking=True
            )
            targets = batch["mask"][matching_indices].to(
                device, non_blocking=True
            )
            probabilities = torch.sigmoid(model(images))

            for local_index, batch_index in enumerate(matching_indices):
                image_id = str(batch["image_id"][batch_index])
                collected[image_id] = convert_sample(
                    image_id,
                    images[local_index],
                    targets[local_index],
                    probabilities[local_index],
                )

        if (
            batch_number == 1
            or batch_number % PROGRESS_INTERVAL == 0
            or batch_number == total_batches
        ):
            print(
                f"Validation [{batch_number}/{total_batches}] "
                f"selected={len(collected)}/{len(selected_ids)}"
            )

        if len(collected) == len(selected_ids):
            break

    missing_ids = selected_ids.difference(collected)
    if missing_ids:
        missing_text = "\n".join(sorted(missing_ids))
        raise ValueError(
            "Selected validation images were not found:\n" + missing_text
        )

    return collected


def add_mask_overlay(axis, image, mask, color, title):
    axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., :3] = color
    overlay[..., 3] = mask.astype(np.float32) * 0.55
    axis.imshow(overlay)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def add_comparison_overlay(axis, sample):
    target = sample["target"]
    prediction = sample["prediction"]
    overlap = np.logical_and(target, prediction)
    missed = np.logical_and(target, ~prediction)
    extra = np.logical_and(~target, prediction)

    overlay = np.zeros((*target.shape, 4), dtype=np.float32)
    overlay[missed] = (0.0, 1.0, 0.0, 0.65)
    overlay[extra] = (1.0, 0.0, 0.0, 0.65)
    overlay[overlap] = (1.0, 1.0, 0.0, 0.75)

    axis.imshow(sample["image"], cmap="gray", vmin=0.0, vmax=1.0)
    axis.imshow(overlay)
    axis.set_title(
        "Comparison\n"
        f"Dice={sample['dice']:.3f}, "
        f"pred={sample['predicted_pixels']:,} px",
        fontsize=9,
    )
    axis.axis("off")


def create_figure(samples, title, output_path):
    figure, axes = plt.subplots(
        len(samples), 4, figsize=(14, 3.45 * len(samples)), squeeze=False
    )

    for row, sample in enumerate(samples):
        short_id = sample["image_id"].split(".")[-4:]
        short_id = ".".join(short_id)

        axes[row, 0].imshow(
            sample["image"], cmap="gray", vmin=0.0, vmax=1.0
        )
        axes[row, 0].set_title(
            f"X-ray\nID ending: {short_id}", fontsize=9
        )
        axes[row, 0].axis("off")

        add_mask_overlay(
            axes[row, 1],
            sample["image"],
            sample["target"],
            (0.0, 1.0, 0.0),
            f"Expert mask (green)\n{sample['target_pixels']:,} px",
        )
        add_mask_overlay(
            axes[row, 2],
            sample["image"],
            sample["prediction"],
            (1.0, 0.0, 0.0),
            "Prediction (red)\n"
            f"max probability={sample['maximum_probability']:.4f}",
        )
        add_comparison_overlay(axes[row, 3], sample)

    figure.suptitle(
        title
        + f"\nValidation only, threshold {PREDICTION_THRESHOLD:.2f}; "
        "yellow=overlap, green=missed, red=incorrect",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_selection_text(hardest_rows, false_positive_rows, samples):
    lines = [
        "Pneumothorax 512 x 512 visual error audit",
        "===========================================",
        f"Checkpoint: {CHECKPOINT_PATH.resolve()}",
        f"Probability threshold: {PREDICTION_THRESHOLD:.2f}",
        "Dataset split: validation only",
        "Test split used: False",
        "Training performed: False",
        "No threshold or post-processing rule was selected.",
        "",
        "Hardest positive localizations",
        "Image ID | Target pixels | Predicted pixels | Dice | Max probability",
    ]

    for row in hardest_rows:
        sample = samples[row["image_id"]]
        lines.append(
            f"{sample['image_id']} | {sample['target_pixels']} | "
            f"{sample['predicted_pixels']} | {sample['dice']:.6f} | "
            f"{sample['maximum_probability']:.6f}"
        )

    lines.extend(
        [
            "",
            "Largest false-positive localizations",
            "Image ID | Predicted pixels | Predicted area | Max probability",
        ]
    )
    for row in false_positive_rows:
        sample = samples[row["image_id"]]
        predicted_area = (
            100.0 * sample["predicted_pixels"] / (IMAGE_SIZE * IMAGE_SIZE)
        )
        lines.append(
            f"{sample['image_id']} | {sample['predicted_pixels']} | "
            f"{predicted_area:.4f}% | {sample['maximum_probability']:.6f}"
        )

    return "\n".join(lines) + "\n"


def main():
    device = choose_device()
    print("Pneumothorax 512 x 512 visual error audit")
    print("-------------------------------------------")
    print(f"Device: {device}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print(f"Probability threshold: {PREDICTION_THRESHOLD:.2f} (provisional)")
    print("Dataset split: validation only")
    print("Test split: not created or accessed")
    print("Training: disabled")

    hardest_rows, false_positive_rows = choose_cases()
    selected_ids = {
        row["image_id"] for row in hardest_rows + false_positive_rows
    }
    print(f"Selected images: {len(selected_ids)}")

    dataset = PneumothoraxDataset(split="validation", image_size=IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False, freeze_encoder=False
    ).to(device)
    checkpoint = load_checkpoint(model, device)
    print(f"Checkpoint epoch: {checkpoint.get('completed_epoch', 'unknown')}")
    print("\nCollecting selected validation cases...")

    samples = collect_selected_samples(model, loader, device, selected_ids)
    hardest_samples = [samples[row["image_id"]] for row in hardest_rows]
    false_positive_samples = [
        samples[row["image_id"]] for row in false_positive_rows
    ]

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    create_figure(
        hardest_samples,
        "Hardest positive pneumothorax localizations",
        HARDEST_POSITIVE_PATH,
    )
    create_figure(
        false_positive_samples,
        "Largest false-positive pneumothorax localizations",
        LARGEST_FALSE_POSITIVE_PATH,
    )
    SELECTION_PATH.write_text(
        build_selection_text(
            hardest_rows, false_positive_rows, samples
        ),
        encoding="utf-8",
    )

    print("\nVisual audit files created")
    print(f"Hard positives: {HARDEST_POSITIVE_PATH.resolve()}")
    print(f"False positives: {LARGEST_FALSE_POSITIVE_PATH.resolve()}")
    print(f"Selection details: {SELECTION_PATH.resolve()}")
    print("No model parameters, thresholds, or masks were changed.")


if __name__ == "__main__":
    main()
