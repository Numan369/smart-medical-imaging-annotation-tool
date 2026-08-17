import os

# Prevent excessive CPU-thread memory use on Windows.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent

PTX_DIRECTORY = PROJECT_ROOT / "PTX-498"

INFERENCE_SCRIPT = (
    PROJECT_ROOT
    / "inference_backend"
    / "pneumothorax_inference.py"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "pneumothorax_512_v4a_fresh_45_55_best.pth"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "ptx498_v4a_random_outputs"
)

EXPECTED_CHECKPOINT_HASH = (
    "109e102e7a521abc6c904e1c5ad214e1a3f18a5b3bfd3dc0d51c98be4a585635"
)


def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def select_image(requested_image=None, seed=None):
    if requested_image is not None:
        image_path = Path(requested_image)

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        image_path = image_path.resolve()

        if not image_path.exists():
            raise FileNotFoundError(
                f"Selected image does not exist: {image_path}"
            )

        return image_path

    image_paths = sorted(
        PTX_DIRECTORY.rglob("*.1.img.png")
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No PTX images were found in {PTX_DIRECTORY}"
        )

    generator = (
        random.Random(seed)
        if seed is not None
        else random.SystemRandom()
    )

    return generator.choice(image_paths)


def get_reference_mask_path(image_path):
    if not image_path.name.endswith(".1.img.png"):
        raise ValueError(
            "Expected an image filename ending in '.1.img.png'."
        )

    mask_name = image_path.name.replace(
        ".1.img.png",
        ".2.mask.png",
    )

    mask_path = image_path.with_name(mask_name)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Matching reference mask was not found: {mask_path}"
        )

    return mask_path


def calculate_metrics(reference_mask, predicted_mask):
    reference = reference_mask.astype(bool)
    predicted = predicted_mask.astype(bool)

    true_positive = int(
        np.logical_and(reference, predicted).sum()
    )

    false_positive = int(
        np.logical_and(~reference, predicted).sum()
    )

    false_negative = int(
        np.logical_and(reference, ~predicted).sum()
    )

    reference_pixels = int(reference.sum())
    predicted_pixels = int(predicted.sum())

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    union = int(
        np.logical_or(reference, predicted).sum()
    )

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2 * true_positive
            / dice_denominator
        )

    if union == 0:
        iou = 1.0
    else:
        iou = true_positive / union

    if predicted_pixels == 0:
        precision = (
            1.0
            if reference_pixels == 0
            else 0.0
        )
    else:
        precision = (
            true_positive / predicted_pixels
        )

    if reference_pixels == 0:
        recall = (
            1.0
            if predicted_pixels == 0
            else 0.0
        )
    else:
        recall = (
            true_positive / reference_pixels
        )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "reference_pixels": reference_pixels,
        "predicted_pixels": predicted_pixels,
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
    }


def create_coloured_overlay(
    grayscale_image,
    mask,
    colour,
    opacity=0.50,
):
    rgb_image = np.repeat(
        grayscale_image[..., None],
        3,
        axis=2,
    )

    colour_array = np.asarray(
        colour,
        dtype=np.float32,
    )

    output = rgb_image.copy()

    output[mask] = (
        (1.0 - opacity) * output[mask]
        + opacity * colour_array
    )

    return np.clip(output, 0.0, 1.0)


def create_comparison_overlay(
    grayscale_image,
    reference_mask,
    predicted_mask,
):
    output = np.repeat(
        grayscale_image[..., None],
        3,
        axis=2,
    )

    reference_only = np.logical_and(
        reference_mask,
        ~predicted_mask,
    )

    predicted_only = np.logical_and(
        predicted_mask,
        ~reference_mask,
    )

    overlap = np.logical_and(
        reference_mask,
        predicted_mask,
    )

    colours = {
        "reference": np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        ),
        "prediction": np.array(
            [0.0, 0.85, 1.0],
            dtype=np.float32,
        ),
        "overlap": np.array(
            [1.0, 0.80, 0.0],
            dtype=np.float32,
        ),
    }

    opacity = 0.60

    for selected_pixels, colour in [
        (reference_only, colours["reference"]),
        (predicted_only, colours["prediction"]),
        (overlap, colours["overlap"]),
    ]:
        output[selected_pixels] = (
            (1.0 - opacity)
            * output[selected_pixels]
            + opacity
            * colour
        )

    return np.clip(output, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen V4A model on a random "
            "or selected PTX-498 image."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help=(
            "Optional path to a specific '*.1.img.png' "
            "file. If omitted, a random image is used."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for repeatable selection."
        ),
    )

    arguments = parser.parse_args()

    for required_path in [
        PTX_DIRECTORY,
        INFERENCE_SCRIPT,
        CHECKPOINT_PATH,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required path does not exist: {required_path}"
            )

    print("VERIFYING FROZEN CHECKPOINT")
    print("---------------------------")

    observed_hash = calculate_sha256(
        CHECKPOINT_PATH
    )

    print(f"Observed SHA-256: {observed_hash}")
    print(
        "Matches frozen checkpoint:",
        observed_hash == EXPECTED_CHECKPOINT_HASH,
    )

    if observed_hash != EXPECTED_CHECKPOINT_HASH:
        raise RuntimeError(
            "Checkpoint hash does not match the frozen V4A model."
        )

    image_path = select_image(
        requested_image=arguments.image,
        seed=arguments.seed,
    )

    reference_mask_path = (
        get_reference_mask_path(image_path)
    )

    case_identifier = image_path.name.replace(
        ".1.img.png",
        "",
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_directory = (
        OUTPUT_ROOT
        / f"case_{case_identifier}_{timestamp}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("\nSELECTED PTX CASE")
    print("-----------------")
    print(f"Case: {case_identifier}")
    print(f"Image: {image_path}")
    print(f"Reference mask: {reference_mask_path}")
    print(f"Output directory: {output_directory}")

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

    print("\nRUNNING FROZEN V4A INFERENCE")
    print("----------------------------")

    completed_process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )

    if completed_process.stdout:
        print(completed_process.stdout)

    if completed_process.stderr:
        print(completed_process.stderr)

    if completed_process.returncode != 0:
        raise RuntimeError(
            "The frozen inference command failed."
        )

    result_files = list(
        output_directory.glob("*_ai_result.json")
    )

    predicted_mask_files = list(
        output_directory.glob("*_ai_mask.png")
    )

    if len(result_files) != 1:
        raise RuntimeError(
            "Exactly one AI result JSON file was expected."
        )

    if len(predicted_mask_files) != 1:
        raise RuntimeError(
            "Exactly one AI mask file was expected."
        )

    result_payload = json.loads(
        result_files[0].read_text(
            encoding="utf-8"
        )
    )

    inference_result = result_payload.get(
        "result",
        result_payload,
    )

    xray_image = Image.open(
        image_path
    ).convert("L")

    reference_image = Image.open(
        reference_mask_path
    ).convert("L")

    predicted_image = Image.open(
        predicted_mask_files[0]
    ).convert("L")

    if reference_image.size != xray_image.size:
        reference_image = reference_image.resize(
            xray_image.size,
            Image.Resampling.NEAREST,
        )

    if predicted_image.size != xray_image.size:
        predicted_image = predicted_image.resize(
            xray_image.size,
            Image.Resampling.NEAREST,
        )

    xray_array = (
        np.asarray(
            xray_image,
            dtype=np.float32,
        )
        / 255.0
    )

    reference_mask = (
        np.asarray(reference_image)
        > 127
    )

    predicted_mask = (
        np.asarray(predicted_image)
        > 127
    )

    metrics = calculate_metrics(
        reference_mask,
        predicted_mask,
    )

    reference_overlay = create_coloured_overlay(
        xray_array,
        reference_mask,
        colour=[0.0, 1.0, 0.0],
    )

    prediction_overlay = create_coloured_overlay(
        xray_array,
        predicted_mask,
        colour=[0.0, 0.85, 1.0],
    )

    comparison_overlay = create_comparison_overlay(
        xray_array,
        reference_mask,
        predicted_mask,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    axes[0, 0].imshow(
        xray_array,
        cmap="gray",
    )
    axes[0, 0].set_title("Original PTX X-ray")

    axes[0, 1].imshow(
        reference_mask,
        cmap="gray",
    )
    axes[0, 1].set_title(
        "Reference mask\nDataset annotation"
    )

    axes[0, 2].imshow(
        predicted_mask,
        cmap="gray",
    )
    axes[0, 2].set_title(
        "Frozen V4A AI mask"
    )

    axes[1, 0].imshow(
        reference_overlay
    )
    axes[1, 0].set_title(
        "Reference overlay — green"
    )

    axes[1, 1].imshow(
        prediction_overlay
    )
    axes[1, 1].set_title(
        "AI overlay — cyan"
    )

    axes[1, 2].imshow(
        comparison_overlay
    )
    axes[1, 2].set_title(
        "Comparison\n"
        "Green: reference only | "
        "Cyan: AI only | "
        "Yellow: overlap"
    )

    for axis in axes.flat:
        axis.axis("off")

    finding_title = inference_result.get(
        "findingTitle",
        inference_result.get(
            "finding",
            "Unknown result",
        ),
    )

    figure.suptitle(
        (
            f"PTX-498 case {case_identifier} — "
            f"{finding_title}\n"
            f"Dice: {metrics['dice']:.4f} | "
            f"IoU: {metrics['iou']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        ),
        fontsize=14,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=[0.0, 0.0, 1.0, 0.93]
    )

    comparison_path = (
        output_directory
        / f"{case_identifier}_v4a_comparison.png"
    )

    figure.savefig(
        comparison_path,
        dpi=180,
        bbox_inches="tight",
    )

    print("\nCASE-SPECIFIC REFERENCE AGREEMENT")
    print("---------------------------------")
    print(f"Finding: {finding_title}")
    print(
        "Region count:",
        inference_result.get("regionCount"),
    )
    print(
        "Mask coverage:",
        inference_result.get(
            "maskCoveragePercent"
        ),
    )
    print(
        "Maximum model output score:",
        inference_result.get(
            "maximumOutputScore"
        ),
    )
    print(f"Reference pixels: {metrics['reference_pixels']}")
    print(f"Predicted pixels: {metrics['predicted_pixels']}")
    print(f"Dice: {metrics['dice']:.6f}")
    print(f"IoU: {metrics['iou']:.6f}")
    print(f"Pixel precision: {metrics['precision']:.6f}")
    print(f"Pixel recall: {metrics['recall']:.6f}")
    print(f"Comparison saved: {comparison_path}")

    print("\nIMPORTANT")
    print("---------")
    print(
        "These metrics apply only to this PTX case "
        "because its reference annotation is available."
    )
    print(
        "They are not overall V4A accuracy and must not "
        "be used to recalibrate the frozen settings."
    )

    plt.show()


if __name__ == "__main__":
    main()