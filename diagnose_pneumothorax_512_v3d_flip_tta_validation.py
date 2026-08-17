"""Validation-only V3D horizontal-flip TTA diagnostic for locked V3C.

Controlled change
-----------------
Standard V3C probabilities are compared with one fixed, training-free method:

    TTA probability = mean(
        probability(original image),
        flip_back(probability(horizontally flipped image)),
    )

Both paths use the locked threshold 0.35. The script never trains, changes a
checkpoint, tunes a threshold, or instantiates the test split. It evaluates
only split="validation" and reports whether flip-TTA changes positive Dice,
positive misses, negative false positives, the joint selection score, and the
viewer-left/viewer-right performance gap.

Expected companion files
------------------------
* pneumothorax_dataset.py
* pneumothorax_model.py
* prepared_data/dataset_splits.csv
* SIIM_TRAIN_TEST/train-rle.csv and referenced DICOM images
* checkpoints/pneumothorax_512_v3c_batchnorm_stabilized_best.pth, or the
  equivalent checkpoint path supplied with --checkpoint
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pneumothorax_dataset import PneumothoraxDataset
from pneumothorax_model import PneumothoraxResNet34UNet


IMAGE_SIZE = 512
BATCH_SIZE = 2
LOCKED_THRESHOLD = 0.35
EXPECTED_STAGE = "pneumothorax_512_v3c_batchnorm_stabilized_finetune"
EXPECTED_EPOCH = 5
EXPECTED_CONTROLLED_CHANGE = (
    "freeze BatchNorm running statistics during V3B training"
)
EXPECTED_BATCHNORM_MODE = "saved_running_statistics_during_training"

LOCAL_CHECKPOINT = (
    Path("checkpoints")
    / "pneumothorax_512_v3c_batchnorm_stabilized_best.pth"
)
DRIVE_PROJECT_DIRECTORY = Path(
    "/content/drive/MyDrive/SmartMedicalImagingColab"
)
DRIVE_CHECKPOINT = (
    DRIVE_PROJECT_DIRECTORY
    / "checkpoints"
    / "pneumothorax_512_v3c_batchnorm_stabilized_best.pth"
)
LOCAL_OUTPUT_DIRECTORY = Path(
    "diagnostics_v3d_flip_tta_validation"
)
DRIVE_OUTPUT_DIRECTORY = (
    DRIVE_PROJECT_DIRECTORY
    / "diagnostics_v3d_flip_tta_validation"
)

LESION_SIZE_BINS = (
    ("tiny", 0.0, 0.001),
    ("small", 0.001, 0.005),
    ("medium", 0.005, 0.02),
    ("large", 0.02, 1.01),
)

CSV_FIELDS = (
    "image_id",
    "is_positive",
    "target_area_pixels",
    "target_area_fraction",
    "image_half",
    "vertical_region",
    "lesion_size_bin",
    "standard_predicted_area_pixels",
    "tta_predicted_area_pixels",
    "standard_dice",
    "tta_dice",
    "dice_delta_tta_minus_standard",
    "standard_max_probability",
    "tta_max_probability",
    "flip_probability_mean_absolute_difference",
    "standard_empty_prediction",
    "tta_empty_prediction",
)


def default_checkpoint_path():
    if DRIVE_CHECKPOINT.is_file():
        return DRIVE_CHECKPOINT
    return LOCAL_CHECKPOINT


def default_output_directory():
    if DRIVE_PROJECT_DIRECTORY.is_dir():
        return DRIVE_OUTPUT_DIRECTORY
    return LOCAL_OUTPUT_DIRECTORY


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Compare locked V3C with fixed horizontal-flip mean TTA on "
            "the SIIM validation split only."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Full V3C epoch-5 best checkpoint. The Colab Drive path is "
            "used automatically when available; otherwise ./checkpoints/."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Output folder. Defaults to Google Drive when mounted, "
            "otherwise a local diagnostics folder."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Process two validation batches to verify the pipeline.",
    )
    return parser.parse_args()


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_torch_checkpoint(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def values_match(actual, expected, tolerance=1e-12):
    try:
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    except (TypeError, ValueError):
        return False


def validate_checkpoint_metadata(checkpoint):
    if checkpoint.get("training_stage") != EXPECTED_STAGE:
        raise ValueError(
            "Expected the full V3C fine-tuning checkpoint; found stage "
            f"{checkpoint.get('training_stage')!r}."
        )
    if checkpoint.get("completed_epoch") != EXPECTED_EPOCH:
        raise ValueError(
            "Expected locked V3C epoch 5; found epoch "
            f"{checkpoint.get('completed_epoch')!r}."
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint has no model_state_dict.")

    configuration = checkpoint.get("configuration", {})
    required = {
        "image_size": IMAGE_SIZE,
        "prediction_threshold": LOCKED_THRESHOLD,
        "controlled_change": EXPECTED_CONTROLLED_CHANGE,
        "batchnorm_mode": EXPECTED_BATCHNORM_MODE,
        "batchnorm_running_statistics_frozen": True,
        "validation_split_used": True,
        "test_split_used": False,
    }
    for name, expected in required.items():
        actual = configuration.get(name)
        matches = (
            values_match(actual, expected)
            if isinstance(expected, float)
            else actual == expected
        )
        if not matches:
            raise ValueError(
                f"Unexpected checkpoint setting {name}: {actual!r}; "
                f"expected {expected!r}."
            )


def load_locked_model(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"V3C checkpoint not found: {checkpoint_path.resolve()}"
        )

    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    validate_checkpoint_metadata(checkpoint)

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def lesion_size_bin(area_fraction):
    for name, lower, upper in LESION_SIZE_BINS:
        if lower <= area_fraction < upper:
            return name
    raise ValueError(f"Unexpected target area fraction: {area_fraction}")


def target_location(binary_target):
    coordinates = torch.nonzero(binary_target, as_tuple=False)
    if coordinates.numel() == 0:
        return "none", "none"

    height, width = binary_target.shape
    centroid_y = float(coordinates[:, 0].float().mean().item()) / height
    centroid_x = float(coordinates[:, 1].float().mean().item()) / width

    image_half = "viewer_left" if centroid_x < 0.5 else "viewer_right"
    if centroid_y < 1.0 / 3.0:
        vertical_region = "upper"
    elif centroid_y < 2.0 / 3.0:
        vertical_region = "middle"
    else:
        vertical_region = "lower"
    return image_half, vertical_region


def binary_dice(prediction, target):
    intersection = int((prediction & target).sum().item())
    predicted_area = int(prediction.sum().item())
    target_area = int(target.sum().item())
    denominator = predicted_area + target_area
    return (
        2.0 * intersection / denominator
        if denominator > 0
        else 1.0
    )


@torch.inference_mode()
def collect_comparison_records(
    model,
    validation_loader,
    device,
    maximum_batches=None,
):
    model.eval()
    records = []
    use_amp = device.type == "cuda"
    total_batches = len(validation_loader)
    if maximum_batches is not None:
        total_batches = min(total_batches, maximum_batches)
    started = time.perf_counter()

    for batch_number, batch in enumerate(validation_loader, start=1):
        if batch_number > total_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        image_ids = batch["image_id"]

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            original_logits = model(images)
            flipped_logits = model(torch.flip(images, dims=(-1,)))

        original_probabilities = torch.sigmoid(original_logits).float()
        flipped_back_probabilities = torch.flip(
            torch.sigmoid(flipped_logits).float(),
            dims=(-1,),
        )
        tta_probabilities = 0.5 * (
            original_probabilities + flipped_back_probabilities
        )

        binary_targets = targets >= 0.5
        standard_predictions = (
            original_probabilities >= LOCKED_THRESHOLD
        )
        tta_predictions = tta_probabilities >= LOCKED_THRESHOLD

        for index in range(images.shape[0]):
            target = binary_targets[index, 0]
            standard_prediction = standard_predictions[index, 0]
            tta_prediction = tta_predictions[index, 0]
            target_area = int(target.sum().item())
            target_area_fraction = target_area / (IMAGE_SIZE * IMAGE_SIZE)
            is_positive = target_area > 0
            image_half, vertical_region = target_location(target)

            standard_area = int(standard_prediction.sum().item())
            tta_area = int(tta_prediction.sum().item())
            standard_dice = binary_dice(
                standard_prediction,
                target,
            )
            tta_dice = binary_dice(tta_prediction, target)
            disagreement = torch.mean(
                torch.abs(
                    original_probabilities[index, 0]
                    - flipped_back_probabilities[index, 0]
                )
            ).item()

            records.append(
                {
                    "image_id": image_ids[index],
                    "is_positive": is_positive,
                    "target_area_pixels": target_area,
                    "target_area_fraction": target_area_fraction,
                    "image_half": image_half,
                    "vertical_region": vertical_region,
                    "lesion_size_bin": (
                        lesion_size_bin(target_area_fraction)
                        if is_positive
                        else "empty"
                    ),
                    "standard_predicted_area_pixels": standard_area,
                    "tta_predicted_area_pixels": tta_area,
                    "standard_dice": standard_dice,
                    "tta_dice": tta_dice,
                    "dice_delta_tta_minus_standard": (
                        tta_dice - standard_dice
                    ),
                    "standard_max_probability": float(
                        original_probabilities[index, 0].max().item()
                    ),
                    "tta_max_probability": float(
                        tta_probabilities[index, 0].max().item()
                    ),
                    "flip_probability_mean_absolute_difference": float(
                        disagreement
                    ),
                    "standard_empty_prediction": standard_area == 0,
                    "tta_empty_prediction": tta_area == 0,
                }
            )

        if (
            batch_number == 1
            or batch_number % 25 == 0
            or batch_number == total_batches
        ):
            elapsed_minutes = (time.perf_counter() - started) / 60.0
            images_done = len(records)
            images_expected = min(
                len(validation_loader.dataset),
                total_batches * validation_loader.batch_size,
            )
            rate = images_done / max(elapsed_minutes, 1e-9)
            remaining_minutes = (
                (images_expected - images_done) / max(rate, 1e-9)
            )
            print(
                f"  V3D comparison [{batch_number}/{total_batches} batches, "
                f"{images_done}/{images_expected} images] "
                f"elapsed={elapsed_minutes:.1f} min, "
                f"est. remaining={remaining_minutes:.1f} min",
                flush=True,
            )

    return records, time.perf_counter() - started


def harmonic_mean(first, second):
    if first <= 0.0 or second <= 0.0:
        return 0.0
    return 2.0 * first * second / (first + second)


def metrics_for_method(records, prefix):
    dice_key = f"{prefix}_dice"
    area_key = f"{prefix}_predicted_area_pixels"
    positive_records = [record for record in records if record["is_positive"]]
    negative_records = [record for record in records if not record["is_positive"]]

    positive_dice = (
        float(np.mean([record[dice_key] for record in positive_records]))
        if positive_records
        else 0.0
    )
    positive_misses = sum(
        record[area_key] == 0 for record in positive_records
    )
    empty_negatives = sum(
        record[area_key] == 0 for record in negative_records
    )
    negative_empty_accuracy = (
        empty_negatives / len(negative_records)
        if negative_records
        else 0.0
    )
    negative_mean_predicted_fraction = (
        float(
            np.mean(
                [
                    record[area_key] / (IMAGE_SIZE * IMAGE_SIZE)
                    for record in negative_records
                ]
            )
        )
        if negative_records
        else 0.0
    )

    return {
        "positive_dice": positive_dice,
        "positive_samples": len(positive_records),
        "empty_positive_predictions": positive_misses,
        "positive_miss_rate": (
            positive_misses / len(positive_records)
            if positive_records
            else 0.0
        ),
        "negative_empty_accuracy": negative_empty_accuracy,
        "negative_samples": len(negative_records),
        "false_positive_negative_images": (
            len(negative_records) - empty_negatives
        ),
        "negative_false_positive_rate": (
            1.0 - negative_empty_accuracy
            if negative_records
            else 0.0
        ),
        "negative_mean_predicted_fraction": (
            negative_mean_predicted_fraction
        ),
        "selection_score": harmonic_mean(
            positive_dice,
            negative_empty_accuracy,
        ),
    }


def grouped_positive_metrics(records, group_key):
    positive_records = [record for record in records if record["is_positive"]]
    result = {}
    for group in sorted({record[group_key] for record in positive_records}):
        bucket = [
            record
            for record in positive_records
            if record[group_key] == group
        ]
        result[group] = {
            "cases": len(bucket),
            "standard_mean_dice": float(
                np.mean([record["standard_dice"] for record in bucket])
            ),
            "tta_mean_dice": float(
                np.mean([record["tta_dice"] for record in bucket])
            ),
            "mean_dice_delta": float(
                np.mean(
                    [
                        record["dice_delta_tta_minus_standard"]
                        for record in bucket
                    ]
                )
            ),
            "standard_misses": sum(
                record["standard_empty_prediction"] for record in bucket
            ),
            "tta_misses": sum(
                record["tta_empty_prediction"] for record in bucket
            ),
        }
    return result


def paired_positive_summary(records):
    deltas = np.asarray(
        [
            record["dice_delta_tta_minus_standard"]
            for record in records
            if record["is_positive"]
        ],
        dtype=np.float64,
    )
    if deltas.size == 0:
        return {
            "improved_cases": 0,
            "worsened_cases": 0,
            "unchanged_cases": 0,
            "mean_delta": 0.0,
            "median_delta": 0.0,
            "bootstrap_mean_delta_95_interval": [0.0, 0.0],
        }

    tolerance = 1e-9
    generator = np.random.default_rng(42)
    sample_indices = generator.integers(
        0,
        deltas.size,
        size=(10000, deltas.size),
    )
    bootstrap_means = deltas[sample_indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "improved_cases": int(np.sum(deltas > tolerance)),
        "worsened_cases": int(np.sum(deltas < -tolerance)),
        "unchanged_cases": int(np.sum(np.abs(deltas) <= tolerance)),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "bootstrap_mean_delta_95_interval": [
            float(lower),
            float(upper),
        ],
    }


def side_gap(side_metrics, method_prefix):
    left = side_metrics.get("viewer_left")
    right = side_metrics.get("viewer_right")
    if left is None or right is None:
        return None
    return (
        left[f"{method_prefix}_mean_dice"]
        - right[f"{method_prefix}_mean_dice"]
    )


def compare_with_checkpoint_reference(checkpoint, standard_metrics):
    reference = checkpoint.get("validation_results", {})
    comparisons = {}
    mapping = {
        "positive_dice": "positive_dice",
        "positive_miss_rate": "positive_miss_rate",
        "negative_empty_accuracy": "negative_empty_accuracy",
        "selection_score": "selection_score",
    }
    for output_name, checkpoint_name in mapping.items():
        expected = reference.get(checkpoint_name)
        measured = standard_metrics[output_name]
        comparisons[output_name] = {
            "checkpoint_value": expected,
            "measured_value": measured,
            "absolute_difference": (
                abs(float(expected) - measured)
                if expected is not None
                else None
            ),
        }
    return comparisons


def build_summary(
    records,
    checkpoint,
    checkpoint_path,
    elapsed_seconds,
    smoke_test,
):
    standard = metrics_for_method(records, "standard")
    tta = metrics_for_method(records, "tta")
    side_metrics = grouped_positive_metrics(records, "image_half")
    summary = {
        "experiment": "V3D validation-only horizontal-flip mean TTA",
        "run_mode": "smoke_test" if smoke_test else "full_validation",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("completed_epoch"),
        "threshold": LOCKED_THRESHOLD,
        "validation_images_processed": len(records),
        "elapsed_seconds": elapsed_seconds,
        "standard_v3c": standard,
        "flip_tta": tta,
        "delta_tta_minus_standard": {
            key: tta[key] - standard[key]
            for key in (
                "positive_dice",
                "positive_miss_rate",
                "negative_empty_accuracy",
                "negative_false_positive_rate",
                "negative_mean_predicted_fraction",
                "selection_score",
            )
        },
        "paired_positive_case_analysis": paired_positive_summary(records),
        "by_image_half": side_metrics,
        "by_lesion_size": grouped_positive_metrics(
            records,
            "lesion_size_bin",
        ),
        "by_vertical_region": grouped_positive_metrics(
            records,
            "vertical_region",
        ),
        "viewer_left_minus_viewer_right_dice_gap": {
            "standard": side_gap(side_metrics, "standard"),
            "flip_tta": side_gap(side_metrics, "tta"),
        },
        "standard_reproduction_check": compare_with_checkpoint_reference(
            checkpoint,
            standard,
        ),
        "protocol": {
            "validation_split_used": True,
            "test_split_used": False,
            "training_performed": False,
            "checkpoint_modified": False,
            "threshold_changed": False,
            "threshold_tuned": False,
            "tta_method": (
                "arithmetic mean of original and horizontally "
                "flipped-back probability maps"
            ),
            "note": (
                "Image-half labels describe pixel coordinates, not "
                "clinical laterality."
            ),
        },
    }
    return summary


def format_method(name, metrics):
    return [
        name,
        f"  Positive Dice: {metrics['positive_dice']:.6f}",
        "  Positive misses: "
        f"{metrics['empty_positive_predictions']} / "
        f"{metrics['positive_samples']} "
        f"({100.0 * metrics['positive_miss_rate']:.2f}%)",
        "  False-positive negatives: "
        f"{metrics['false_positive_negative_images']} / "
        f"{metrics['negative_samples']} "
        f"({100.0 * metrics['negative_false_positive_rate']:.2f}%)",
        "  Negative empty-mask accuracy: "
        f"{metrics['negative_empty_accuracy']:.6f}",
        "  Mean predicted negative area: "
        f"{100.0 * metrics['negative_mean_predicted_fraction']:.6f}%",
        f"  Joint selection score: {metrics['selection_score']:.6f}",
    ]


def create_text_report(summary):
    standard = summary["standard_v3c"]
    tta = summary["flip_tta"]
    delta = summary["delta_tta_minus_standard"]
    paired = summary["paired_positive_case_analysis"]
    lines = [
        "V3D VALIDATION-ONLY FLIP-TTA DIAGNOSTIC",
        "========================================",
        f"Run mode: {summary['run_mode']}",
        f"Validation images processed: {summary['validation_images_processed']}",
        f"Locked checkpoint epoch: {summary['checkpoint_epoch']}",
        f"Locked threshold: {summary['threshold']:.2f}",
        "Test split instantiated: No",
        "",
    ]
    lines.extend(format_method("STANDARD V3C", standard))
    lines.append("")
    lines.extend(format_method("V3D FLIP-TTA", tta))
    lines.extend(
        [
            "",
            "DELTA: TTA MINUS STANDARD",
            "-------------------------",
            f"  Positive Dice: {delta['positive_dice']:+.6f}",
            "  Positive miss rate: "
            f"{100.0 * delta['positive_miss_rate']:+.2f} percentage points",
            "  Negative false-positive rate: "
            f"{100.0 * delta['negative_false_positive_rate']:+.2f} "
            "percentage points",
            "  Negative empty accuracy: "
            f"{delta['negative_empty_accuracy']:+.6f}",
            f"  Joint selection score: {delta['selection_score']:+.6f}",
            "",
            "PAIRED POSITIVE-CASE ANALYSIS",
            "-----------------------------",
            f"  Improved cases: {paired['improved_cases']}",
            f"  Worsened cases: {paired['worsened_cases']}",
            f"  Unchanged cases: {paired['unchanged_cases']}",
            f"  Mean Dice change: {paired['mean_delta']:+.6f}",
            f"  Median Dice change: {paired['median_delta']:+.6f}",
            "  Descriptive bootstrap 95% interval for mean change: "
            f"[{paired['bootstrap_mean_delta_95_interval'][0]:+.6f}, "
            f"{paired['bootstrap_mean_delta_95_interval'][1]:+.6f}]",
            "",
            "IMAGE-HALF RESULTS (PIXEL COORDINATES)",
            "--------------------------------------",
        ]
    )

    for group, metrics in summary["by_image_half"].items():
        lines.append(
            f"  {group}: n={metrics['cases']} | "
            f"standard={metrics['standard_mean_dice']:.6f} | "
            f"TTA={metrics['tta_mean_dice']:.6f} | "
            f"delta={metrics['mean_dice_delta']:+.6f} | "
            f"misses {metrics['standard_misses']} -> {metrics['tta_misses']}"
        )

    gaps = summary["viewer_left_minus_viewer_right_dice_gap"]
    if gaps["standard"] is not None and gaps["flip_tta"] is not None:
        lines.extend(
            [
                "  Viewer-left minus viewer-right Dice gap:",
                f"    standard={gaps['standard']:+.6f}",
                f"    TTA={gaps['flip_tta']:+.6f}",
            ]
        )

    lines.extend(
        [
            "",
            "LESION-SIZE RESULTS",
            "-------------------",
        ]
    )
    for group, metrics in summary["by_lesion_size"].items():
        lines.append(
            f"  {group}: n={metrics['cases']} | "
            f"standard={metrics['standard_mean_dice']:.6f} | "
            f"TTA={metrics['tta_mean_dice']:.6f} | "
            f"delta={metrics['mean_dice_delta']:+.6f} | "
            f"misses {metrics['standard_misses']} -> {metrics['tta_misses']}"
        )

    lines.extend(
        [
            "",
            "This was a report-only validation diagnostic.",
            "No training, threshold tuning, checkpoint change, or test-split "
            "access occurred.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_arguments()
    checkpoint_path = (
        args.checkpoint.expanduser()
        if args.checkpoint is not None
        else default_checkpoint_path()
    )
    output_directory = (
        args.output_directory.expanduser()
        if args.output_directory is not None
        else default_output_directory()
    )
    if args.smoke_test:
        output_directory = output_directory.with_name(
            output_directory.name + "_smoke"
        )

    device = choose_device()
    print("V3D validation-only horizontal-flip TTA diagnostic")
    print("--------------------------------------------------")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Run mode: {'SMOKE TEST' if args.smoke_test else 'FULL VALIDATION'}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Fixed threshold: {LOCKED_THRESHOLD}")
    print("Controlled change: mean horizontal-flip TTA only")
    print("Validation split only; test split will not be instantiated")
    print("No training or checkpoint modification\n")

    model, checkpoint = load_locked_model(checkpoint_path, device)
    print(
        f"Loaded V3C checkpoint epoch {checkpoint['completed_epoch']} "
        f"({checkpoint['training_stage']})"
    )

    validation_dataset = PneumothoraxDataset(
        split="validation",
        image_size=IMAGE_SIZE,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Validation images: {len(validation_dataset)} "
        "(test split NOT instantiated)"
    )

    records, elapsed_seconds = collect_comparison_records(
        model=model,
        validation_loader=validation_loader,
        device=device,
        maximum_batches=2 if args.smoke_test else None,
    )
    summary = build_summary(
        records=records,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        elapsed_seconds=elapsed_seconds,
        smoke_test=args.smoke_test,
    )
    report = create_text_report(summary)

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "v3d_flip_tta_per_case.csv"
    json_path = output_directory / "v3d_flip_tta_summary.json"
    text_path = output_directory / "v3d_flip_tta_summary.txt"
    write_csv(csv_path, records)
    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    text_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"Per-case CSV: {csv_path.resolve()}")
    print(f"Summary JSON: {json_path.resolve()}")
    print(f"Summary text: {text_path.resolve()}")
    print("Test split was never created or accessed.")


if __name__ == "__main__":
    main()
