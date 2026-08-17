"""Evaluate the locked V3C model on every available PTX-498 PNG pair.

This is an external, report-only evaluation. PTX-498 masks are never used for
training, checkpoint selection, threshold tuning, or model modification. The
script loads the slim V3C deployment checkpoint once and processes one image at
a time to keep CPU memory use low.

Outputs
-------
* ptx498_v3c_per_case_metrics.csv
* ptx498_v3c_summary.json
* ptx498_v3c_summary.txt
* representative_grids/*.png
* predicted_masks/*.png (optional; enabled by default)

Progress is checkpointed every five cases. If the run is interrupted, the
partial CSV is retained and a later run skips already completed cases.
"""

import os

for variable_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable_name] = "1"

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image, ImageDraw

import compare_ptx498_v3c_random as comparison
import infer_single_pneumothorax_v3c as inference


DEFAULT_DATASET_DIRECTORY = Path("PTX-498")
DEFAULT_OUTPUT_DIRECTORY = Path("ptx498_v3c_full_evaluation")
PARTIAL_FILENAME = "ptx498_v3c_metrics_partial.csv"
FINAL_CSV_FILENAME = "ptx498_v3c_per_case_metrics.csv"
SUMMARY_JSON_FILENAME = "ptx498_v3c_summary.json"
SUMMARY_TEXT_FILENAME = "ptx498_v3c_summary.txt"
CHECKPOINT_INTERVAL = 5

CSV_FIELDS = (
    "case_id",
    "image_path",
    "mask_path",
    "predicted_mask_path",
    "height",
    "width",
    "ground_truth_pixels",
    "predicted_pixels",
    "overlap_pixels",
    "ground_truth_area_fraction",
    "predicted_area_fraction",
    "dice",
    "precision",
    "recall",
    "iou",
    "empty_ground_truth",
    "empty_prediction",
    "image_half",
    "vertical_region",
    "lesion_size_bin",
)


def parse_arguments():
    parser = ArgumentParser(
        description=(
            "Evaluate the locked V3C epoch-5 model on all complete "
            "PTX-498 PNG image/mask pairs."
        )
    )
    parser.add_argument(
        "dataset_directory",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_DIRECTORY,
        help="PTX-498 folder; defaults to ./PTX-498.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=inference.CHECKPOINT_PATH,
        help=(
            "Slim V3C deployment checkpoint; defaults to "
            "./checkpoints/pneumothorax_512_v3c_epoch5_deployment.pth."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for CSV, summary, masks and representative grids.",
    )
    parser.add_argument(
        "--no-save-masks",
        action="store_true",
        help="Do not retain every predicted binary PNG mask.",
    )
    parser.add_argument(
        "--no-grids",
        action="store_true",
        help="Do not generate representative visual grids.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore any partial CSV and evaluate every pair again.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional smoke-test limit; omit for the full evaluation.",
    )
    return parser.parse_args()


def case_id_for(image_path):
    return comparison.safe_case_name(image_path)


def lesion_size_bin(area_fraction):
    """Fixed descriptive bins based on fraction of the full X-ray."""

    if area_fraction <= 0.0:
        return "empty"
    if area_fraction < 0.005:
        return "tiny_lt_0.5pct"
    if area_fraction < 0.02:
        return "small_0.5_to_2pct"
    if area_fraction < 0.05:
        return "medium_2_to_5pct"
    return "large_ge_5pct"


def mask_location(mask):
    """Describe mask centroid in image coordinates without assuming anatomy."""

    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return "none", "none"

    centroid_x = float(columns.mean()) / mask.shape[1]
    centroid_y = float(rows.mean()) / mask.shape[0]
    image_half = "viewer_left" if centroid_x < 0.5 else "viewer_right"

    if centroid_y < 1.0 / 3.0:
        vertical_region = "upper"
    elif centroid_y < 2.0 / 3.0:
        vertical_region = "middle"
    else:
        vertical_region = "lower"
    return image_half, vertical_region


def record_for(image_path, mask_path, image, actual, predicted, metrics):
    pixel_count = int(actual.size)
    ground_truth_fraction = metrics["actual_pixels"] / pixel_count
    predicted_fraction = metrics["predicted_pixels"] / pixel_count
    image_half, vertical_region = mask_location(actual)

    return {
        "case_id": case_id_for(image_path),
        "image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "height": int(actual.shape[0]),
        "width": int(actual.shape[1]),
        "ground_truth_pixels": metrics["actual_pixels"],
        "predicted_pixels": metrics["predicted_pixels"],
        "overlap_pixels": metrics["overlap_pixels"],
        "ground_truth_area_fraction": ground_truth_fraction,
        "predicted_area_fraction": predicted_fraction,
        "dice": metrics["dice"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "iou": metrics["iou"],
        "empty_ground_truth": metrics["actual_pixels"] == 0,
        "empty_prediction": metrics["predicted_pixels"] == 0,
        "image_half": image_half,
        "vertical_region": vertical_region,
        "lesion_size_bin": lesion_size_bin(ground_truth_fraction),
    }


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(path)


def read_partial_csv(path):
    if not path.is_file():
        return []

    integer_fields = {
        "height",
        "width",
        "ground_truth_pixels",
        "predicted_pixels",
        "overlap_pixels",
    }
    float_fields = {
        "ground_truth_area_fraction",
        "predicted_area_fraction",
        "dice",
        "precision",
        "recall",
        "iou",
    }
    boolean_fields = {"empty_ground_truth", "empty_prediction"}

    records = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for field in integer_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            for field in boolean_fields:
                row[field] = row[field].lower() == "true"
            records.append(row)
    return records


def metric_summary(records, field):
    values = [float(record[field]) for record in records]
    if not values:
        return None
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def bootstrap_mean_interval(records, field, repetitions=5000, seed=42):
    """Return a deterministic descriptive case-level bootstrap interval."""

    values = np.asarray([float(record[field]) for record in records])
    if values.size == 0:
        return None
    if values.size == 1:
        value = float(values[0])
        return {"lower_95": value, "upper_95": value}

    generator = np.random.default_rng(seed)
    sample_indices = generator.integers(
        0,
        values.size,
        size=(repetitions, values.size),
    )
    sample_means = values[sample_indices].mean(axis=1)
    lower, upper = np.percentile(sample_means, [2.5, 97.5])
    return {"lower_95": float(lower), "upper_95": float(upper)}


def grouped_summary(records, field):
    result = {}
    for group in sorted({record[field] for record in records}):
        group_records = [record for record in records if record[field] == group]
        result[group] = {
            "cases": len(group_records),
            "dice": metric_summary(group_records, "dice"),
            "recall": metric_summary(group_records, "recall"),
            "empty_prediction_count": sum(
                bool(record["empty_prediction"]) for record in group_records
            ),
        }
    return result


def build_summary(records, pair_count, elapsed_seconds, checkpoint_path):
    positive_records = [
        record for record in records if not record["empty_ground_truth"]
    ]
    negative_records = [
        record for record in records if record["empty_ground_truth"]
    ]
    missed_positive_count = sum(
        bool(record["empty_prediction"]) for record in positive_records
    )

    summary = {
        "evaluation_name": "PTX-498 external V3C report-only evaluation",
        "checkpoint": str(checkpoint_path.resolve()),
        "locked_completed_epoch": inference.EXPECTED_COMPLETED_EPOCH,
        "locked_threshold": inference.EXPECTED_THRESHOLD,
        "model_input_size": inference.EXPECTED_IMAGE_SIZE,
        "discovered_complete_pairs": pair_count,
        "evaluated_cases": len(records),
        "positive_ground_truth_cases": len(positive_records),
        "empty_ground_truth_cases": len(negative_records),
        "runtime_seconds_this_invocation": elapsed_seconds,
        "metrics_all_cases": {
            field: metric_summary(records, field)
            for field in ("dice", "precision", "recall", "iou")
        },
        "mean_dice_descriptive_bootstrap_95_interval": (
            bootstrap_mean_interval(records, "dice")
        ),
        "complete_misses_on_positive_cases": missed_positive_count,
        "complete_miss_rate_on_positive_cases": (
            missed_positive_count / len(positive_records)
            if positive_records
            else None
        ),
        "dice_at_least_0.70_count": sum(
            float(record["dice"]) >= 0.70 for record in records
        ),
        "dice_below_0.20_count": sum(
            float(record["dice"]) < 0.20 for record in records
        ),
        "by_ground_truth_size": grouped_summary(records, "lesion_size_bin"),
        "by_image_half": grouped_summary(positive_records, "image_half"),
        "by_vertical_region": grouped_summary(
            positive_records,
            "vertical_region",
        ),
        "protocol": {
            "external_masks_used_for_training": False,
            "external_masks_used_for_checkpoint_selection": False,
            "external_masks_used_for_threshold_tuning": False,
            "test_time_augmentation": False,
            "threshold_changed": False,
            "note": (
                "Image-half groups use pixel coordinates (viewer_left and "
                "viewer_right); they do not infer clinical laterality."
            ),
        },
    }
    return summary


def format_metric_line(name, metric):
    return (
        f"{name:<10} mean={metric['mean']:.4f}  "
        f"median={metric['median']:.4f}  "
        f"min={metric['minimum']:.4f}  max={metric['maximum']:.4f}"
    )


def summary_text(summary):
    lines = [
        "PTX-498 EXTERNAL V3C EVALUATION",
        "================================",
        f"Evaluated cases: {summary['evaluated_cases']} / "
        f"{summary['discovered_complete_pairs']}",
        f"Positive-mask cases: {summary['positive_ground_truth_cases']}",
        f"Empty-mask cases: {summary['empty_ground_truth_cases']}",
        f"Locked checkpoint epoch: {summary['locked_completed_epoch']}",
        f"Locked threshold: {summary['locked_threshold']:.2f}",
        "",
        "OVERALL METRICS",
        "---------------",
    ]
    for name, metric in summary["metrics_all_cases"].items():
        lines.append(format_metric_line(name.capitalize(), metric))

    interval = summary["mean_dice_descriptive_bootstrap_95_interval"]
    lines.extend(
        [
            "",
            "FAILURE COUNTS",
            "--------------",
            "Complete positive-case misses: "
            f"{summary['complete_misses_on_positive_cases']} "
            f"({100.0 * (summary['complete_miss_rate_on_positive_cases'] or 0):.1f}%)",
            f"Cases with Dice >= 0.70: {summary['dice_at_least_0.70_count']}",
            f"Cases with Dice < 0.20: {summary['dice_below_0.20_count']}",
            "Mean Dice descriptive bootstrap 95% interval: "
            f"[{interval['lower_95']:.4f}, {interval['upper_95']:.4f}]",
            "",
            "GROUPED DICE",
            "------------",
        ]
    )

    for heading, key in (
        ("Ground-truth lesion size", "by_ground_truth_size"),
        ("Image half", "by_image_half"),
        ("Vertical region", "by_vertical_region"),
    ):
        lines.append(heading + ":")
        for group, values in summary[key].items():
            dice = values["dice"]
            lines.append(
                f"  {group:<22} n={values['cases']:<3} "
                f"mean Dice={dice['mean']:.4f}  "
                f"median={dice['median']:.4f}  "
                f"empty predictions={values['empty_prediction_count']}"
            )

    lines.extend(
        [
            "",
            "External masks were used for reporting only.",
            "The checkpoint and threshold were not modified.",
            "Image-half labels describe displayed pixel coordinates, not "
            "clinical laterality.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_case_arrays(record):
    image_path = Path(record["image_path"])
    mask_path = Path(record["mask_path"])
    image, _, _, _, _, _ = inference.prepare_image(image_path)
    actual = comparison.load_ground_truth(mask_path, image.shape)
    predicted_mask_path = (
        Path(record["predicted_mask_path"])
        if record.get("predicted_mask_path")
        else None
    )
    if predicted_mask_path and predicted_mask_path.is_file():
        with Image.open(predicted_mask_path) as mask_image:
            predicted = np.asarray(mask_image.convert("L")) > 0
    else:
        predicted = None
    return image, actual, predicted


def compact_case_row(image, actual, predicted, record, panel_side=260):
    base = comparison.normalized_rgb(image)
    if predicted is None:
        raise FileNotFoundError(
            "Representative grids require saved prediction masks."
        )

    panels = (
        base,
        comparison.mask_overlay(base, actual, [255, 215, 0]),
        comparison.mask_overlay(base, predicted, [0, 210, 255]),
        comparison.error_comparison(base, actual, predicted),
    )
    resized = []
    for panel in panels:
        pil_panel = Image.fromarray(panel)
        scale = panel_side / max(pil_panel.size)
        size = (
            max(1, round(pil_panel.width * scale)),
            max(1, round(pil_panel.height * scale)),
        )
        resized.append(
            pil_panel.resize(size, resample=Image.Resampling.BILINEAR)
        )

    panel_width = max(panel.width for panel in resized)
    panel_height = max(panel.height for panel in resized)
    title_height = 42
    gap = 5
    row = Image.new(
        "RGB",
        (panel_width * 4 + gap * 3, panel_height + title_height),
        "white",
    )
    draw = ImageDraw.Draw(row)
    title = (
        f"{record['case_id']} | Dice {float(record['dice']):.4f} | "
        f"P {float(record['precision']):.4f} | "
        f"R {float(record['recall']):.4f}"
    )
    draw.text((7, 7), title, fill="black")
    for index, panel in enumerate(resized):
        left = index * (panel_width + gap)
        top = title_height + (panel_height - panel.height) // 2
        row.paste(panel, (left, top))
    return row


def save_grid(path, title, selected_records):
    if not selected_records:
        return None
    rows = []
    for record in selected_records:
        image, actual, predicted = load_case_arrays(record)
        rows.append(compact_case_row(image, actual, predicted, record))

    title_height = 45
    gap = 8
    width = max(row.width for row in rows)
    height = title_height + sum(row.height for row in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 12), title, fill="black")
    top = title_height
    for row in rows:
        canvas.paste(row, (0, top))
        top += row.height + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def representative_groups(records, count=4):
    ordered = sorted(records, key=lambda record: float(record["dice"]))
    misses = [record for record in ordered if record["empty_prediction"]]
    nonempty = [record for record in ordered if not record["empty_prediction"]]

    middle = []
    if ordered:
        centre = len(ordered) // 2
        start = max(0, centre - count // 2)
        middle = ordered[start : start + count]

    return {
        "complete_misses": misses[:count],
        "lowest_nonempty": nonempty[:count],
        "median_cases": middle,
        "best_cases": list(reversed(ordered[-count:])),
    }


def main():
    args = parse_arguments()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    inference.configure_torch_cpu()
    dataset_directory = args.dataset_directory.expanduser()
    output_directory = args.output_directory.expanduser()
    checkpoint_path = args.checkpoint.expanduser()
    inference.CHECKPOINT_PATH = checkpoint_path

    pairs = comparison.discover_pairs(dataset_directory)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    output_directory.mkdir(parents=True, exist_ok=True)
    partial_path = output_directory / PARTIAL_FILENAME
    if args.restart:
        records = []
    else:
        records = read_partial_csv(partial_path)
    completed_ids = {record["case_id"] for record in records}

    device = inference.choose_device()
    print("PTX-498 full external V3C evaluation")
    print("---------------------------------------")
    print(f"Device: {device}")
    print(f"Complete pairs selected: {len(pairs)}")
    print(f"Already completed from partial CSV: {len(completed_ids)}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Locked threshold: {inference.EXPECTED_THRESHOLD}")
    print("External masks: reporting only; no tuning or training")

    model, checkpoint = inference.load_locked_model(device)
    print(f"Loaded locked V3C checkpoint epoch {checkpoint['completed_epoch']}")

    masks_directory = output_directory / "predicted_masks"
    if not args.no_save_masks:
        masks_directory.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    newly_completed = 0
    try:
        for pair_index, (image_path, mask_path) in enumerate(pairs, start=1):
            case_id = case_id_for(image_path)
            if case_id in completed_ids:
                print(f"[{pair_index}/{len(pairs)}] {case_id}: already completed")
                continue

            image, image_tensor, height, width, source_type, _ = (
                inference.prepare_image(image_path)
            )
            if source_type != "PNG":
                raise ValueError(f"Expected PNG input: {image_path}")
            actual = comparison.load_ground_truth(mask_path, image.shape)
            predicted = inference.predict_mask(
                model,
                image_tensor,
                device,
                output_size=(height, width),
            )
            metrics = comparison.calculate_metrics(predicted, actual)
            record = record_for(
                image_path,
                mask_path,
                image,
                actual,
                predicted,
                metrics,
            )

            if not args.no_save_masks:
                predicted_mask_path = masks_directory / f"{case_id}_v3c_mask.png"
                Image.fromarray(predicted.astype(np.uint8) * 255).save(
                    predicted_mask_path
                )
                record["predicted_mask_path"] = str(predicted_mask_path.resolve())

            records.append(record)
            completed_ids.add(case_id)
            newly_completed += 1
            print(
                f"[{pair_index}/{len(pairs)}] {case_id}: "
                f"Dice={metrics['dice']:.4f} "
                f"P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f}"
            )

            if newly_completed % CHECKPOINT_INTERVAL == 0:
                write_csv(partial_path, records)
                print(f"  Progress saved: {len(records)} cases")
    except KeyboardInterrupt:
        write_csv(partial_path, records)
        print("\nInterrupted. Partial results were saved safely.")
        raise SystemExit(130)
    except Exception:
        write_csv(partial_path, records)
        print("\nAn error occurred. Partial results were saved safely.")
        raise

    write_csv(partial_path, records)
    expected_case_ids = {case_id_for(image_path) for image_path, _ in pairs}
    records = [
        record for record in records if record["case_id"] in expected_case_ids
    ]
    records.sort(key=lambda record: record["case_id"])

    final_csv_path = output_directory / FINAL_CSV_FILENAME
    write_csv(final_csv_path, records)
    elapsed_seconds = time.perf_counter() - start_time
    summary = build_summary(
        records,
        pair_count=len(pairs),
        elapsed_seconds=elapsed_seconds,
        checkpoint_path=checkpoint_path,
    )

    summary_json_path = output_directory / SUMMARY_JSON_FILENAME
    summary_json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    summary_text_path = output_directory / SUMMARY_TEXT_FILENAME
    report = summary_text(summary)
    summary_text_path.write_text(report, encoding="utf-8")

    grid_paths = []
    if not args.no_grids:
        if args.no_save_masks:
            print("Representative grids skipped because --no-save-masks was used.")
        else:
            grids_directory = output_directory / "representative_grids"
            for group_name, selected_records in representative_groups(records).items():
                grid_path = save_grid(
                    grids_directory / f"{group_name}.png",
                    group_name.replace("_", " ").title(),
                    selected_records,
                )
                if grid_path is not None:
                    grid_paths.append(grid_path)

    print("\n" + report)
    print(f"Per-case CSV: {final_csv_path.resolve()}")
    print(f"Summary JSON: {summary_json_path.resolve()}")
    print(f"Summary text: {summary_text_path.resolve()}")
    for grid_path in grid_paths:
        print(f"Representative grid: {grid_path.resolve()}")
    print("The model, checkpoint, and threshold were not modified.")


if __name__ == "__main__":
    main()
