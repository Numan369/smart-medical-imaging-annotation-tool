"""Low-memory, resumable local V3D flip-TTA validation diagnostic.

This Windows/CPU script compares the locked V3C epoch-5 prediction with one
fixed test-time augmentation method:

    TTA probability = mean(
        probability(original image),
        flip_back(probability(horizontally flipped image)),
    )

Safety and reproducibility properties:
* Uses the slim, memory-mapped V3C deployment checkpoint.
* Imports the lean deployment model from infer_single_pneumothorax_v3c.py;
  torchvision and the full training checkpoint are not used.
* Processes one validation image and one forward pass at a time.
* Saves atomic partial results every 10 new images.
* Automatically resumes and verifies saved image IDs after interruption.
* Instantiates split="validation" only. The test split is never created.
* Performs no training, threshold tuning, or checkpoint modification.
"""

import os

for variable_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable_name] = "1"

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

import infer_single_pneumothorax_v3c as inference
from pneumothorax_dataset import PneumothoraxDataset


IMAGE_SIZE = 512
LOCKED_THRESHOLD = 0.35
SAVE_INTERVAL = 10
EXPECTED_VALIDATION_IMAGES = 1205
EXPECTED_VALIDATION_POSITIVES = 267
EXPECTED_VALIDATION_NEGATIVES = 938

DEFAULT_CHECKPOINT = (
    Path("checkpoints")
    / "pneumothorax_512_v3c_epoch5_deployment.pth"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "diagnostics_v3d_flip_tta_validation_local"
)

REFERENCE_V3C = {
    "positive_dice": 0.3845211124816763,
    "empty_positive_predictions": 35,
    "positive_miss_rate": 0.13108614232209737,
    "false_positive_negative_images": 240,
    "negative_empty_accuracy": 0.744136460554371,
    "negative_mean_predicted_fraction": 0.0006318814210546042,
    "selection_score": 0.5070380715753282,
}

LESION_SIZE_BINS = (
    ("tiny", 0.0, 0.001),
    ("small", 0.001, 0.005),
    ("medium", 0.005, 0.02),
    ("large", 0.02, 1.01),
)

CSV_FIELDS = (
    "dataset_index",
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
    "seconds_for_case",
)

INTEGER_FIELDS = {
    "dataset_index",
    "target_area_pixels",
    "standard_predicted_area_pixels",
    "tta_predicted_area_pixels",
}
FLOAT_FIELDS = {
    "target_area_fraction",
    "standard_dice",
    "tta_dice",
    "dice_delta_tta_minus_standard",
    "standard_max_probability",
    "tta_max_probability",
    "flip_probability_mean_absolute_difference",
    "seconds_for_case",
}
BOOLEAN_FIELDS = {
    "is_positive",
    "standard_empty_prediction",
    "tta_empty_prediction",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run low-memory, resumable V3C versus flip-TTA comparison "
            "on the local SIIM validation split only."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Slim V3C epoch-5 deployment checkpoint.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Folder for partial progress and final reports.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Process four validation images in a separate smoke folder.",
    )
    return parser.parse_args()


def validate_local_protocol():
    if inference.EXPECTED_COMPLETED_EPOCH != 5:
        raise ValueError("Lean inference module is not locked to epoch 5.")
    if inference.EXPECTED_IMAGE_SIZE != IMAGE_SIZE:
        raise ValueError("Lean inference module is not locked to 512 x 512.")
    if not math.isclose(
        inference.EXPECTED_THRESHOLD,
        LOCKED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Lean inference module threshold is not 0.35.")


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
    if denominator == 0:
        return 1.0
    return 2.0 * intersection / denominator


@torch.inference_mode()
def evaluate_one_case(model, sample, dataset_index):
    case_started = time.perf_counter()
    image = sample["image"].unsqueeze(0)
    target = sample["mask"][0] >= 0.5

    original_logits = model(image)
    original_probability = torch.sigmoid(original_logits)[0, 0]
    del original_logits

    flipped_image = torch.flip(image, dims=(-1,))
    flipped_logits = model(flipped_image)
    flipped_back_probability = torch.flip(
        torch.sigmoid(flipped_logits)[0, 0],
        dims=(-1,),
    )
    del flipped_logits, flipped_image

    tta_probability = 0.5 * (
        original_probability + flipped_back_probability
    )
    standard_prediction = original_probability >= LOCKED_THRESHOLD
    tta_prediction = tta_probability >= LOCKED_THRESHOLD

    target_area = int(target.sum().item())
    target_area_fraction = target_area / (IMAGE_SIZE * IMAGE_SIZE)
    is_positive = target_area > 0
    standard_area = int(standard_prediction.sum().item())
    tta_area = int(tta_prediction.sum().item())
    standard_dice = binary_dice(standard_prediction, target)
    tta_dice = binary_dice(tta_prediction, target)
    image_half, vertical_region = target_location(target)
    disagreement = float(
        torch.mean(
            torch.abs(
                original_probability - flipped_back_probability
            )
        ).item()
    )

    record = {
        "dataset_index": dataset_index,
        "image_id": sample["image_id"],
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
        "dice_delta_tta_minus_standard": tta_dice - standard_dice,
        "standard_max_probability": float(
            original_probability.max().item()
        ),
        "tta_max_probability": float(tta_probability.max().item()),
        "flip_probability_mean_absolute_difference": disagreement,
        "standard_empty_prediction": standard_area == 0,
        "tta_empty_prediction": tta_area == 0,
        "seconds_for_case": time.perf_counter() - case_started,
    }

    del (
        image,
        target,
        original_probability,
        flipped_back_probability,
        tta_probability,
        standard_prediction,
        tta_prediction,
    )
    return record


def atomic_write_text(path, text):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2))


def atomic_write_csv(path, records):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(path)


def read_partial_csv(path):
    if not path.is_file():
        return []

    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(
                "Partial CSV columns do not match this evaluator version. "
                "Rename the existing output folder and start a clean run."
            )
        for row in reader:
            for field in INTEGER_FIELDS:
                row[field] = int(row[field])
            for field in FLOAT_FIELDS:
                row[field] = float(row[field])
            for field in BOOLEAN_FIELDS:
                row[field] = row[field].lower() == "true"
            records.append(row)
    return records


def checkpoint_identity(path):
    resolved = path.resolve()
    status = resolved.stat()
    return {
        "resolved_path": str(resolved),
        "size_bytes": status.st_size,
        "modified_time_ns": status.st_mtime_ns,
        "completed_epoch": inference.EXPECTED_COMPLETED_EPOCH,
        "threshold": LOCKED_THRESHOLD,
    }


def expected_manifest(checkpoint_path, dataset, smoke_test):
    return {
        "experiment": "local_resumable_v3d_flip_tta_validation",
        "checkpoint": checkpoint_identity(checkpoint_path),
        "dataset_split": "validation",
        "dataset_images": len(dataset),
        "image_size": IMAGE_SIZE,
        "threshold": LOCKED_THRESHOLD,
        "tta_method": (
            "mean of original and horizontally flipped-back "
            "probability maps"
        ),
        "save_interval_images": SAVE_INTERVAL,
        "smoke_test": smoke_test,
        "test_split_used": False,
    }


def prepare_manifest(path, expected):
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                "Existing progress belongs to a different checkpoint, "
                "dataset, or protocol. Rename the output directory before "
                "starting a new experiment."
            )
    else:
        atomic_write_json(path, expected)


def validate_resumed_records(records, dataset, maximum_images):
    seen_indices = set()
    for record in records:
        index = record["dataset_index"]
        if index in seen_indices:
            raise ValueError(f"Duplicate resumed dataset index: {index}")
        if index < 0 or index >= maximum_images:
            raise ValueError(f"Resumed index is outside this run: {index}")
        expected_id = dataset.rows[index]["ImageId"]
        if record["image_id"] != expected_id:
            raise ValueError(
                f"Resumed image ID mismatch at index {index}: "
                f"{record['image_id']} versus {expected_id}."
            )
        seen_indices.add(index)
    return seen_indices


def harmonic_mean(first, second):
    if first <= 0.0 or second <= 0.0:
        return 0.0
    return 2.0 * first * second / (first + second)


def metrics_for_method(records, prefix):
    dice_key = f"{prefix}_dice"
    area_key = f"{prefix}_predicted_area_pixels"
    positives = [record for record in records if record["is_positive"]]
    negatives = [record for record in records if not record["is_positive"]]

    positive_dice = (
        statistics.fmean(record[dice_key] for record in positives)
        if positives
        else 0.0
    )
    positive_misses = sum(record[area_key] == 0 for record in positives)
    empty_negatives = sum(record[area_key] == 0 for record in negatives)
    negative_empty_accuracy = (
        empty_negatives / len(negatives) if negatives else 0.0
    )
    negative_mean_predicted_fraction = (
        statistics.fmean(
            record[area_key] / (IMAGE_SIZE * IMAGE_SIZE)
            for record in negatives
        )
        if negatives
        else 0.0
    )
    return {
        "positive_dice": positive_dice,
        "positive_samples": len(positives),
        "empty_positive_predictions": positive_misses,
        "positive_miss_rate": (
            positive_misses / len(positives) if positives else 0.0
        ),
        "negative_empty_accuracy": negative_empty_accuracy,
        "negative_samples": len(negatives),
        "false_positive_negative_images": len(negatives) - empty_negatives,
        "negative_false_positive_rate": (
            1.0 - negative_empty_accuracy if negatives else 0.0
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
    positives = [record for record in records if record["is_positive"]]
    result = {}
    for group in sorted({record[group_key] for record in positives}):
        bucket = [record for record in positives if record[group_key] == group]
        result[group] = {
            "cases": len(bucket),
            "standard_mean_dice": statistics.fmean(
                record["standard_dice"] for record in bucket
            ),
            "tta_mean_dice": statistics.fmean(
                record["tta_dice"] for record in bucket
            ),
            "mean_dice_delta": statistics.fmean(
                record["dice_delta_tta_minus_standard"]
                for record in bucket
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

    generator = np.random.default_rng(42)
    sample_indices = generator.integers(
        0,
        deltas.size,
        size=(10000, deltas.size),
    )
    bootstrap_means = deltas[sample_indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    tolerance = 1e-9
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


def side_gap(side_metrics, prefix):
    left = side_metrics.get("viewer_left")
    right = side_metrics.get("viewer_right")
    if left is None or right is None:
        return None
    return left[f"{prefix}_mean_dice"] - right[f"{prefix}_mean_dice"]


def reproduction_check(standard_metrics):
    result = {}
    for name, expected in REFERENCE_V3C.items():
        measured = standard_metrics[name]
        difference = (
            abs(measured - expected)
            if isinstance(expected, float)
            else abs(int(measured) - expected)
        )
        result[name] = {
            "reference": expected,
            "measured": measured,
            "absolute_difference": difference,
        }
    return result


def build_summary(records, run_seconds, resumed_cases):
    standard = metrics_for_method(records, "standard")
    tta = metrics_for_method(records, "tta")
    side_metrics = grouped_positive_metrics(records, "image_half")
    return {
        "experiment": "local resumable V3D horizontal-flip mean TTA",
        "validation_images": len(records),
        "resumed_cases_at_start": resumed_cases,
        "runtime_seconds_this_invocation": run_seconds,
        "checkpoint_epoch": 5,
        "threshold": LOCKED_THRESHOLD,
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
        "standard_v3c_reproduction_check": reproduction_check(standard),
        "protocol": {
            "validation_split_used": True,
            "test_split_used": False,
            "training_performed": False,
            "checkpoint_modified": False,
            "threshold_changed": False,
            "threshold_tuned": False,
            "checkpoint_format": "slim memory-mapped deployment weights",
            "batch_size": 1,
            "tta_method": (
                "mean of original and horizontally flipped-back "
                "probability maps"
            ),
            "note": (
                "Image-half labels describe pixel coordinates, not "
                "clinical laterality."
            ),
        },
    }


def format_method(title, metrics):
    return [
        title,
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
        "LOCAL V3D VALIDATION-ONLY FLIP-TTA DIAGNOSTIC",
        "================================================",
        f"Validation images: {summary['validation_images']}",
        f"Resumed cases at this start: {summary['resumed_cases_at_start']}",
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
            "  Descriptive bootstrap 95% interval: "
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

    lines.extend(["", "LESION-SIZE RESULTS", "-------------------"])
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
            "No training, threshold tuning, checkpoint modification, or "
            "test-split access occurred.",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_complete_dataset(dataset):
    if len(dataset) != EXPECTED_VALIDATION_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_IMAGES} validation images; "
            f"found {len(dataset)}."
        )
    positive_count = sum(
        int(row["HasPneumothorax"]) for row in dataset.rows
    )
    negative_count = len(dataset) - positive_count
    if positive_count != EXPECTED_VALIDATION_POSITIVES:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_POSITIVES} positives; "
            f"found {positive_count}."
        )
    if negative_count != EXPECTED_VALIDATION_NEGATIVES:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_NEGATIVES} negatives; "
            f"found {negative_count}."
        )


def main():
    args = parse_arguments()
    validate_local_protocol()
    inference.configure_torch_cpu()
    device = torch.device("cpu")

    checkpoint_path = args.checkpoint.expanduser()
    output_directory = args.output_directory.expanduser()
    if args.smoke_test:
        output_directory = output_directory.with_name(
            output_directory.name + "_smoke"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    partial_path = output_directory / "v3d_flip_tta_partial.csv"
    manifest_path = output_directory / "v3d_flip_tta_manifest.json"
    final_csv_path = output_directory / "v3d_flip_tta_per_case.csv"
    summary_json_path = output_directory / "v3d_flip_tta_summary.json"
    summary_text_path = output_directory / "v3d_flip_tta_summary.txt"

    print("Local low-memory V3D flip-TTA validation diagnostic")
    print("---------------------------------------------------")
    print("Device: cpu")
    print(f"Run mode: {'SMOKE TEST' if args.smoke_test else 'FULL VALIDATION'}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Fixed threshold: {LOCKED_THRESHOLD}")
    print("Batch size: 1; original and flip passes are sequential")
    print(f"Progress interval: every {SAVE_INTERVAL} new images")
    print("Validation split only; test split will not be instantiated")
    print("No training or checkpoint modification\n")

    validation_dataset = PneumothoraxDataset(
        split="validation",
        image_size=IMAGE_SIZE,
    )
    validate_complete_dataset(validation_dataset)
    maximum_images = 4 if args.smoke_test else len(validation_dataset)
    print(
        f"Validation dataset verified: {len(validation_dataset)} images "
        f"({EXPECTED_VALIDATION_POSITIVES} positive / "
        f"{EXPECTED_VALIDATION_NEGATIVES} negative)"
    )
    print(f"Images selected for this run: {maximum_images}")

    inference.CHECKPOINT_PATH = checkpoint_path
    model, checkpoint = inference.load_locked_model(device)
    print(
        f"Loaded slim V3C checkpoint epoch "
        f"{checkpoint['completed_epoch']} using memory mapping"
    )
    del checkpoint
    gc.collect()

    manifest = expected_manifest(
        checkpoint_path,
        validation_dataset,
        args.smoke_test,
    )
    prepare_manifest(manifest_path, manifest)
    records = read_partial_csv(partial_path)
    completed_indices = validate_resumed_records(
        records,
        validation_dataset,
        maximum_images,
    )
    resumed_cases = len(completed_indices)
    print(f"Previously completed cases: {resumed_cases}")

    run_started = time.perf_counter()
    newly_completed = 0
    try:
        for dataset_index in range(maximum_images):
            if dataset_index in completed_indices:
                continue

            sample = validation_dataset[dataset_index]
            record = evaluate_one_case(model, sample, dataset_index)
            records.append(record)
            completed_indices.add(dataset_index)
            newly_completed += 1
            del sample

            completed_total = len(completed_indices)
            elapsed = time.perf_counter() - run_started
            rate = newly_completed / max(elapsed, 1e-9)
            remaining = maximum_images - completed_total
            remaining_minutes = remaining / max(rate, 1e-9) / 60.0
            print(
                f"  [{completed_total}/{maximum_images}] "
                f"{record['image_id'][:24]} | "
                f"standard Dice={record['standard_dice']:.4f} | "
                f"TTA Dice={record['tta_dice']:.4f} | "
                f"case={record['seconds_for_case']:.1f}s | "
                f"est. remaining={remaining_minutes:.1f} min",
                flush=True,
            )

            if newly_completed % SAVE_INTERVAL == 0:
                records.sort(key=lambda item: item["dataset_index"])
                atomic_write_csv(partial_path, records)
                gc.collect()
                print(
                    f"    Progress saved safely: {completed_total} cases",
                    flush=True,
                )
    except KeyboardInterrupt:
        records.sort(key=lambda item: item["dataset_index"])
        atomic_write_csv(partial_path, records)
        print("\nInterrupted. Current progress was saved safely.")
        raise SystemExit(130)
    except Exception:
        records.sort(key=lambda item: item["dataset_index"])
        atomic_write_csv(partial_path, records)
        print("\nAn error occurred. Current progress was saved safely.")
        raise

    records.sort(key=lambda item: item["dataset_index"])
    atomic_write_csv(partial_path, records)

    if len(records) != maximum_images:
        raise RuntimeError(
            f"Expected {maximum_images} completed records; found "
            f"{len(records)}."
        )

    run_seconds = time.perf_counter() - run_started
    summary = build_summary(records, run_seconds, resumed_cases)
    report = create_text_report(summary)
    atomic_write_csv(final_csv_path, records)
    atomic_write_json(summary_json_path, summary)
    atomic_write_text(summary_text_path, report)

    print("\n" + report)
    print(f"Per-case CSV: {final_csv_path.resolve()}")
    print(f"Summary JSON: {summary_json_path.resolve()}")
    print(f"Summary text: {summary_text_path.resolve()}")
    print("Test split was never created or accessed.")


if __name__ == "__main__":
    main()
