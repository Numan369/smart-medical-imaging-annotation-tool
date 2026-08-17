"""Random qualitative/metric comparison of V3C on one PTX-498 PNG pair.

Each run selects one matching *.img.png / *.mask.png pair unless an exact
image is supplied. The locked V3C deployment model generates a suggested mask.
The script saves and displays:

1. Original external X-ray
2. Expert ground-truth overlay
3. V3C suggested-mask overlay
4. Error comparison: green overlap, red missed, blue extra

PTX-498 remains an independent external demonstration dataset. Its masks are
used only for reporting and visualization, never for training, threshold
tuning, checkpoint selection or model modification.
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
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageDraw

import infer_single_pneumothorax_v3c as inference


DEFAULT_DATASET_DIRECTORY = Path("PTX-498")
OUTPUT_DIRECTORY = Path("ptx498_v3c_random_outputs")
IMAGE_ENDING = ".img.png"
MASK_ENDING = ".mask.png"


def parse_arguments():
    parser = ArgumentParser(
        description=(
            "Randomly select one PTX-498 PNG pair and compare its expert "
            "mask with the locked V3C suggestion."
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
        "--image",
        type=Path,
        help="Optional exact *.img.png file instead of random selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional reproducible random seed.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Save the preview without opening the system image viewer.",
    )
    return parser.parse_args()


def mask_path_for(image_path):
    """Map 1.1.img.png to 1.2.mask.png and equivalent PTX names."""

    name = image_path.name
    if not name.endswith(IMAGE_ENDING):
        raise ValueError(
            f"Expected a filename ending in {IMAGE_ENDING}: {image_path}"
        )

    prefix = name[: -len(IMAGE_ENDING)]
    parts = prefix.rsplit(".", 1)
    if len(parts) == 2 and parts[1].isdigit():
        mask_name = f"{parts[0]}.{int(parts[1]) + 1}{MASK_ENDING}"
    else:
        mask_name = f"{prefix}{MASK_ENDING}"
    return image_path.with_name(mask_name)


def discover_pairs(directory):
    """Find PNG image/mask pairs while ignoring merges and NIfTI files."""

    if not directory.is_dir():
        raise FileNotFoundError(
            f"PTX-498 directory was not found: {directory.resolve()}"
        )

    pairs = []
    missing_masks = []
    for image_path in sorted(directory.rglob(f"*{IMAGE_ENDING}")):
        mask_path = mask_path_for(image_path)
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
        else:
            missing_masks.append(image_path)

    if missing_masks:
        examples = "\n".join(
            f"  {path}" for path in missing_masks[:5]
        )
        raise FileNotFoundError(
            f"{len(missing_masks)} PTX image(s) have no matching mask. "
            f"Examples:\n{examples}"
        )

    if not pairs:
        raise FileNotFoundError(
            "No matching *.img.png and *.mask.png pairs were found."
        )
    return pairs


def choose_pair(args, pairs):
    """Select an explicit pair or one random complete pair."""

    if args.image is not None:
        image_path = args.image.expanduser()
        if not image_path.is_absolute():
            candidate = args.dataset_directory / image_path
            if candidate.is_file():
                image_path = candidate

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Requested PTX image was not found: {image_path.resolve()}"
            )

        mask_path = mask_path_for(image_path)
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Matching PTX mask was not found: {mask_path.resolve()}"
            )
        return image_path, mask_path

    generator = (
        random.Random(args.seed)
        if args.seed is not None
        else random.SystemRandom()
    )
    return generator.choice(pairs)


def load_ground_truth(mask_path, expected_shape):
    """Load the external expert mask without resizing or smoothing."""

    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0

    if mask.shape != expected_shape:
        raise ValueError(
            f"PTX image/mask shapes differ: {expected_shape} versus "
            f"{mask.shape}."
        )
    return mask


def calculate_metrics(prediction, actual):
    """Calculate foreground-only segmentation metrics."""

    prediction = prediction.astype(bool)
    actual = actual.astype(bool)

    overlap = prediction & actual
    intersection = int(overlap.sum())
    predicted_pixels = int(prediction.sum())
    actual_pixels = int(actual.sum())
    union_pixels = int((prediction | actual).sum())

    dice_denominator = predicted_pixels + actual_pixels
    dice = (
        2.0 * intersection / dice_denominator
        if dice_denominator
        else 1.0
    )
    precision = (
        intersection / predicted_pixels
        if predicted_pixels
        else 0.0
    )
    recall = (
        intersection / actual_pixels
        if actual_pixels
        else 0.0
    )
    iou = intersection / union_pixels if union_pixels else 1.0

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "actual_pixels": actual_pixels,
        "predicted_pixels": predicted_pixels,
        "overlap_pixels": intersection,
    }


def normalized_rgb(image):
    """Convert the normalized grayscale image to uint8 RGB."""

    grayscale = np.round(
        np.clip(image, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    return np.repeat(grayscale[..., None], 3, axis=2)


def mask_overlay(base_rgb, mask, colour, alpha=0.55):
    """Overlay one mask on a grayscale RGB image."""

    result = base_rgb.copy()
    if mask.any():
        colour_array = np.asarray(colour, dtype=np.float32)
        result[mask] = np.round(
            (1.0 - alpha) * result[mask].astype(np.float32)
            + alpha * colour_array
        ).astype(np.uint8)
    return result


def error_comparison(base_rgb, actual, predicted):
    """Encode overlap/misses/extras as green/red/blue."""

    result = base_rgb.copy()
    overlap = actual & predicted
    missed = actual & ~predicted
    extra = ~actual & predicted

    for region, colour in (
        (overlap, [0, 255, 0]),
        (missed, [255, 0, 0]),
        (extra, [0, 100, 255]),
    ):
        if region.any():
            result[region] = np.round(
                0.45 * result[region].astype(np.float32)
                + 0.55 * np.asarray(colour, dtype=np.float32)
            ).astype(np.uint8)
    return result


def resize_panel(panel, maximum_side=560):
    """Create a display-sized panel while retaining the saved full masks."""

    image = Image.fromarray(panel)
    width, height = image.size
    scale = min(1.0, maximum_side / max(width, height))
    size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    return image.resize(size, resample=Image.Resampling.BILINEAR)


def safe_case_name(image_path):
    """Return a path-safe case identifier."""

    name = image_path.name[: -len(IMAGE_ENDING)]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "ptx498_case"


def create_preview(image, actual, predicted, result, image_path):
    """Build a labelled four-panel external comparison."""

    base = normalized_rgb(image)
    panels = (
        resize_panel(base),
        resize_panel(mask_overlay(base, actual, [255, 215, 0])),
        resize_panel(mask_overlay(base, predicted, [0, 210, 255])),
        resize_panel(error_comparison(base, actual, predicted)),
    )
    titles = (
        "Original PTX-498 X-ray",
        "Expert ground truth",
        f"V3C suggestion (threshold {inference.EXPECTED_THRESHOLD:.2f})",
        "Green overlap | Red missed | Blue extra",
    )

    panel_width, panel_height = panels[0].size
    title_height = 40
    summary_height = 52
    gap = 8
    canvas = Image.new(
        "RGB",
        (
            panel_width * 4 + gap * 3,
            panel_height + title_height + summary_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    summary = (
        f"{image_path.name}  |  Dice {result['dice']:.4f}  |  "
        f"Precision {result['precision']:.4f}  |  "
        f"Recall {result['recall']:.4f}  |  IoU {result['iou']:.4f}"
    )
    draw.text((10, 10), summary, fill="black")

    panel_top = summary_height + title_height
    for index, (title, panel) in enumerate(zip(titles, panels)):
        left = index * (panel_width + gap)
        draw.text(
            (left + 7, summary_height + 10),
            title,
            fill="black",
        )
        canvas.paste(panel, (left, panel_top))
    return canvas


def save_outputs(image_path, predicted, preview):
    """Save the model mask and labelled comparison without overwriting PTX."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    case_name = safe_case_name(image_path)
    mask_path = OUTPUT_DIRECTORY / f"{case_name}_v3c_mask.png"
    preview_path = OUTPUT_DIRECTORY / f"{case_name}_v3c_comparison.png"

    Image.fromarray(predicted.astype(np.uint8) * 255).save(mask_path)
    preview.save(preview_path)
    return mask_path, preview_path


def main():
    args = parse_arguments()
    inference.configure_torch_cpu()

    dataset_directory = args.dataset_directory.expanduser()
    pairs = discover_pairs(dataset_directory)
    image_path, mask_path = choose_pair(args, pairs)

    (
        image,
        image_tensor,
        height,
        width,
        source_type,
        converted,
    ) = inference.prepare_image(image_path)
    if source_type != "PNG":
        raise ValueError("PTX-498 comparison expected a PNG image.")
    if converted:
        print("Input note: the PTX PNG was converted to grayscale.")

    actual = load_ground_truth(mask_path, image.shape)
    device = inference.choose_device()
    model, checkpoint = inference.load_locked_model(device)
    predicted = inference.predict_mask(
        model,
        image_tensor,
        device,
        output_size=(height, width),
    )

    result = calculate_metrics(predicted, actual)
    preview = create_preview(
        image,
        actual,
        predicted,
        result,
        image_path,
    )
    prediction_path, preview_path = save_outputs(
        image_path,
        predicted,
        preview,
    )

    print("Random PTX-498 external V3C comparison")
    print("--------------------------------------")
    print(f"Discovered complete pairs: {len(pairs)}")
    print(f"Selected X-ray: {image_path.resolve()}")
    print(f"Expert mask: {mask_path.resolve()}")
    print(f"Device: {device}")
    print(
        "Checkpoint: locked V3C epoch "
        f"{checkpoint['completed_epoch']} deployment weights"
    )
    print(f"Fixed threshold: {inference.EXPECTED_THRESHOLD}")
    print(f"Dice: {result['dice']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"IoU: {result['iou']:.4f}")
    print(f"Actual mask pixels: {result['actual_pixels']:,}")
    print(f"Predicted mask pixels: {result['predicted_pixels']:,}")
    print(f"Overlap pixels: {result['overlap_pixels']:,}")
    print(f"Saved model mask: {prediction_path.resolve()}")
    print(f"Saved comparison: {preview_path.resolve()}")
    print(
        "External masks were used only for reporting; "
        "the model and threshold were not modified."
    )

    if not args.no_open:
        preview.show(
            title="Random PTX-498 V3C comparison — external reporting only"
        )


if __name__ == "__main__":
    main()
