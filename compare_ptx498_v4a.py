import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

PTX_DIRECTORY = PROJECT_ROOT / "PTX-498"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "pneumothorax_512_v4a_fresh_45_55_best.pth"
)

INFERENCE_SCRIPT = (
    PROJECT_ROOT
    / "inference_backend"
    / "pneumothorax_inference.py"
)

OUTPUT_ROOT = PROJECT_ROOT / "ptx498_v4a_test_outputs"


# ---------------------------------------------------------
# Select a PTX-498 case
# ---------------------------------------------------------

def find_ptx_images():
    images = sorted(PTX_DIRECTORY.rglob("*.1.img.png"))

    if not images:
        raise FileNotFoundError(
            f"No PTX X-rays matching '*.1.img.png' were found in:\n"
            f"{PTX_DIRECTORY}"
        )

    return images


def select_image(requested_image=None, seed=None):
    if requested_image:
        image_path = Path(requested_image)

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        image_path = image_path.resolve()

        if not image_path.exists():
            raise FileNotFoundError(
                f"The selected X-ray does not exist:\n{image_path}"
            )

        return image_path

    images = find_ptx_images()

    if seed is not None:
        random.seed(seed)

    return random.choice(images)


def find_reference_mask(image_path):
    image_name = image_path.name

    if not image_name.endswith(".1.img.png"):
        raise ValueError(
            "The selected PTX image must end with '.1.img.png'."
        )

    case_number = image_name.removesuffix(".1.img.png")
    reference_path = image_path.with_name(
        f"{case_number}.2.mask.png"
    )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"The matching actual mask was not found:\n{reference_path}"
        )

    return reference_path, case_number


# ---------------------------------------------------------
# Load images and masks
# ---------------------------------------------------------

def load_xray(path):
    return np.asarray(Image.open(path).convert("L"))


def load_mask(path):
    mask = np.asarray(Image.open(path).convert("L"))
    return mask > 127


def resize_binary_mask(mask, target_shape):
    target_height, target_width = target_shape

    resized = Image.fromarray(
        (mask.astype(np.uint8) * 255)
    ).resize(
        (target_width, target_height),
        resample=Image.Resampling.NEAREST,
    )

    return np.asarray(resized) > 127


# ---------------------------------------------------------
# Calculate comparison metrics
# ---------------------------------------------------------

def calculate_metrics(reference_mask, predicted_mask):
    reference = reference_mask.astype(bool)
    predicted = predicted_mask.astype(bool)

    true_positive = np.logical_and(reference, predicted).sum()
    false_positive = np.logical_and(~reference, predicted).sum()
    false_negative = np.logical_and(reference, ~predicted).sum()

    reference_pixels = reference.sum()
    predicted_pixels = predicted.sum()
    union_pixels = np.logical_or(reference, predicted).sum()

    dice_denominator = reference_pixels + predicted_pixels
    dice = (
        (2.0 * true_positive) / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    iou = (
        true_positive / union_pixels
        if union_pixels > 0
        else 1.0
    )

    precision_denominator = true_positive + false_positive
    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall_denominator = true_positive + false_negative
    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


# ---------------------------------------------------------
# Create overlap visualization
# ---------------------------------------------------------

def create_overlap_image(xray, reference_mask, predicted_mask):
    normalized_xray = xray.astype(np.float32)

    minimum = normalized_xray.min()
    maximum = normalized_xray.max()

    if maximum > minimum:
        normalized_xray = (
            normalized_xray - minimum
        ) / (maximum - minimum)
    else:
        normalized_xray = np.zeros_like(
            normalized_xray,
            dtype=np.float32,
        )

    rgb = np.stack(
        [normalized_xray, normalized_xray, normalized_xray],
        axis=-1,
    )

    reference_only = np.logical_and(
        reference_mask,
        ~predicted_mask,
    )

    predicted_only = np.logical_and(
        predicted_mask,
        ~reference_mask,
    )

    agreement = np.logical_and(
        reference_mask,
        predicted_mask,
    )

    # Green = actual mask only
    rgb[reference_only] = [0.0, 1.0, 0.0]

    # Cyan = AI prediction only
    rgb[predicted_only] = [0.0, 1.0, 1.0]

    # Yellow = actual and AI overlap
    rgb[agreement] = [1.0, 1.0, 0.0]

    return rgb


# ---------------------------------------------------------
# Run V4A prediction
# ---------------------------------------------------------

def run_v4a_inference(image_path, output_directory):
    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        str(image_path),
        "--checkpoint",
        str(CHECKPOINT_PATH),
        "--output-directory",
        str(output_directory),
        "--device",
        "cpu",
    ]

    print()
    print("Running the trained V4A model...")
    print(f"Selected X-ray: {image_path.name}")

    completed = subprocess.run(
        command,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "The V4A model could not generate a prediction."
        )


def find_generated_files(output_directory):
    result_files = sorted(
        output_directory.glob("*_ai_result.json")
    )

    mask_files = sorted(
        output_directory.glob("*_ai_mask.png")
    )

    if not result_files:
        raise FileNotFoundError(
            "The model did not create an AI result JSON file."
        )

    if not mask_files:
        raise FileNotFoundError(
            "The model did not create an AI mask PNG file."
        )

    return result_files[-1], mask_files[-1]


# ---------------------------------------------------------
# Display the four requested panels
# ---------------------------------------------------------

def display_results(
    xray,
    reference_mask,
    predicted_mask,
    overlap_image,
    case_number,
    metrics,
    result,
    save_path,
):
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(18, 6),
    )

    axes[0].imshow(xray, cmap="gray")
    axes[0].set_title("1. X-ray")

    axes[1].imshow(reference_mask, cmap="gray")
    axes[1].set_title("2. Original actual mask")

    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title("3. AI-predicted mask")

    axes[3].imshow(overlap_image)
    axes[3].set_title(
        "4. Overlap\n"
        "Green: actual only | Cyan: AI only | Yellow: agreement"
    )

    for axis in axes:
        axis.axis("off")

    finding_title = result.get(
        "findingTitle",
        result.get("finding", "Prediction generated"),
    )

    region_count = result.get("regionCount", "N/A")
    coverage = result.get("maskCoveragePercent")

    coverage_text = (
        f"{coverage:.4f}%"
        if isinstance(coverage, (int, float))
        else "N/A"
    )

    figure.suptitle(
        f"PTX-498 case {case_number} — {finding_title}\n"
        f"Dice: {metrics['dice']:.4f} | "
        f"IoU: {metrics['iou']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"Regions: {region_count} | "
        f"Coverage: {coverage_text}",
        fontsize=12,
    )

    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(
        save_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test the frozen V4A pneumothorax model on one "
            "PTX-498 image and display four panels."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help=(
            "Optional PTX X-ray path. If omitted, a random "
            "PTX-498 image is selected."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed. Use the same seed to "
            "select the same random image again."
        ),
    )

    args = parser.parse_args()

    if not PTX_DIRECTORY.exists():
        raise FileNotFoundError(
            f"PTX-498 folder was not found:\n{PTX_DIRECTORY}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"V4A checkpoint was not found:\n{CHECKPOINT_PATH}"
        )

    if not INFERENCE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Inference script was not found:\n{INFERENCE_SCRIPT}"
        )

    selected_image = select_image(
        requested_image=args.image,
        seed=args.seed,
    )

    reference_path, case_number = find_reference_mask(
        selected_image
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_directory = (
        OUTPUT_ROOT
        / f"case_{case_number}_{timestamp}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    run_v4a_inference(
        selected_image,
        output_directory,
    )

    result_path, prediction_path = find_generated_files(
        output_directory
    )

    with result_path.open("r", encoding="utf-8") as file:
        complete_result = json.load(file)

    result = complete_result.get(
        "result",
        complete_result,
    )

    xray = load_xray(selected_image)
    reference_mask = load_mask(reference_path)
    predicted_mask = load_mask(prediction_path)

    if reference_mask.shape != xray.shape:
        reference_mask = resize_binary_mask(
            reference_mask,
            xray.shape,
        )

    if predicted_mask.shape != xray.shape:
        predicted_mask = resize_binary_mask(
            predicted_mask,
            xray.shape,
        )

    metrics = calculate_metrics(
        reference_mask,
        predicted_mask,
    )

    overlap_image = create_overlap_image(
        xray,
        reference_mask,
        predicted_mask,
    )

    comparison_path = (
        output_directory
        / f"{case_number}_v4a_four_panel_comparison.png"
    )

    print()
    print("Prediction completed.")
    print(f"Actual mask: {reference_path.name}")
    print(f"AI mask: {prediction_path.name}")
    print(f"Dice: {metrics['dice']:.6f}")
    print(f"IoU: {metrics['iou']:.6f}")
    print(f"Pixel precision: {metrics['precision']:.6f}")
    print(f"Pixel recall: {metrics['recall']:.6f}")
    print(f"Saved comparison: {comparison_path}")
    print()
    print(
        "Colours: green = actual only, cyan = AI only, "
        "yellow = agreement."
    )

    display_results(
        xray=xray,
        reference_mask=reference_mask,
        predicted_mask=predicted_mask,
        overlap_image=overlap_image,
        case_number=case_number,
        metrics=metrics,
        result=result,
        save_path=comparison_path,
    )


if __name__ == "__main__":
    main()