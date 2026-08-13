"""Generate an editable pneumothorax mask for one DICOM chest X-ray."""

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pydicom
import torch
import torch.nn.functional as F

from pneumothorax_dataset import normalise_dicom_image
from pneumothorax_model import PneumothoraxResNet34UNet


CHECKPOINT_PATH = Path("checkpoints") / "fine_tune_stage_best.pth"
OUTPUT_DIRECTORY = Path("inference_outputs")
PREDICTION_THRESHOLD = 0.35
MODEL_IMAGE_SIZE = 256
EXPECTED_TRAINING_STAGE = (
    "balanced_weighted_partial_encoder_finetune"
)


def parse_arguments():
    """Read an optional DICOM path from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a pneumothorax mask for one DICOM chest X-ray. "
            "If no path is supplied, a file-selection window opens."
        )
    )
    parser.add_argument(
        "dicom_path",
        nargs="?",
        type=Path,
        help="Optional path to the DICOM file.",
    )
    return parser.parse_args()


def choose_dicom_file(command_line_path):
    """Use the supplied path or ask the user to choose a DICOM file."""

    if command_line_path is not None:
        path = command_line_path
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as error:
            raise RuntimeError(
                "The file-selection window is unavailable. Run the script "
                "with a DICOM path, for example: "
                "python infer_single_pneumothorax.py image.dcm"
            ) from error

        root = tk.Tk()
        root.withdraw()
        root.update()
        selected_path = filedialog.askopenfilename(
            title="Choose one chest X-ray DICOM file",
            filetypes=(
                ("DICOM files", "*.dcm"),
                ("All files", "*.*"),
            ),
        )
        root.destroy()

        if not selected_path:
            raise SystemExit("No DICOM file was selected.")

        path = Path(selected_path)

    path = path.expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"The selected DICOM file was not found: {path.resolve()}"
        )

    return path


def choose_device():
    """Use an NVIDIA GPU when available, otherwise use the CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_checkpoint(model, device):
    """Restore the validated fine-tuned model weights."""

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "The fine-tuned checkpoint was not found at: "
            f"{CHECKPOINT_PATH.resolve()}"
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain model_state_dict."
        )

    training_stage = checkpoint.get("training_stage")

    if training_stage != EXPECTED_TRAINING_STAGE:
        raise ValueError(
            "Unexpected checkpoint training stage: "
            f"{training_stage!r}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint


def convert_colour_image_to_grayscale(image):
    """Convert an RGB/RGBA DICOM pixel array to one grayscale channel."""

    if image.ndim != 3:
        return image, False

    # pydicom normally returns colour images as (height, width, channels).
    # The second layout is accepted for robustness with unusual converters.
    if image.shape[-1] in (3, 4):
        colour_image = image[..., :3]
    elif image.shape[0] in (3, 4):
        colour_image = np.moveaxis(image[:3], 0, -1)
    else:
        raise ValueError(
            "Expected a grayscale or RGB chest X-ray, but the DICOM "
            f"pixel array has shape {image.shape}."
        )

    # ITU-R BT.601 luminance weights. A true grayscale X-ray stored as RGB
    # has equal channels, so this also reproduces the original gray values.
    grayscale_image = np.sum(
        colour_image.astype(np.float32)
        * np.array([0.299, 0.587, 0.114], dtype=np.float32),
        axis=-1,
    )

    return np.ascontiguousarray(grayscale_image), True


def prepare_image(dicom_path):
    """Read and normalize one DICOM using the training-time method."""

    dicom_data = pydicom.dcmread(dicom_path)
    image = normalise_dicom_image(dicom_data)
    image, converted_from_colour = convert_colour_image_to_grayscale(image)

    if image.ndim != 2:
        raise ValueError(
            "Expected one two-dimensional chest X-ray, but the DICOM "
            f"pixel array has shape {image.shape}."
        )

    original_height, original_width = image.shape
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    resized_tensor = F.interpolate(
        image_tensor,
        size=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return (
        image,
        resized_tensor,
        original_height,
        original_width,
        converted_from_colour,
    )


@torch.no_grad()
def predict_mask(model, image_tensor, device, output_size):
    """Predict at 256 x 256, then restore the original image size."""

    model.eval()
    image_tensor = image_tensor.to(device)
    probabilities = torch.sigmoid(model(image_tensor))
    small_mask = probabilities >= PREDICTION_THRESHOLD

    full_size_mask = F.interpolate(
        small_mask.float(),
        size=output_size,
        mode="nearest",
    )

    return (
        probabilities.squeeze().cpu().numpy(),
        full_size_mask.squeeze().cpu().numpy().astype(bool),
    )


def safe_output_stem(dicom_path):
    """Create a filesystem-safe name for generated files."""

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", dicom_path.stem).strip("._")
    return stem or "dicom_prediction"


def save_results(dicom_path, image, mask):
    """Save an editable binary mask and a visual comparison image."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_stem = safe_output_stem(dicom_path)
    mask_path = OUTPUT_DIRECTORY / f"{output_stem}_mask.png"
    preview_path = OUTPUT_DIRECTORY / f"{output_stem}_preview.png"

    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original DICOM X-ray")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"Suggested mask (threshold {PREDICTION_THRESHOLD:.2f})"
    )

    axes[2].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    overlay = np.ma.masked_where(~mask, mask)
    axes[2].imshow(
        overlay,
        cmap="autumn",
        vmin=0,
        vmax=1,
        alpha=0.45,
        interpolation="nearest",
    )
    axes[2].set_title("Suggested mask overlay")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        "AI-assisted pneumothorax annotation — human review required",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(preview_path, dpi=180, bbox_inches="tight")

    return mask_path, preview_path, figure


def main():
    args = parse_arguments()
    dicom_path = choose_dicom_file(args.dicom_path)
    device = choose_device()

    print("Single-X-ray pneumothorax inference")
    print("----------------------------------")
    print(f"Device: {device}")
    print(f"Selected DICOM: {dicom_path.resolve()}")
    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print(f"Prediction threshold: {PREDICTION_THRESHOLD}")

    (
        image,
        image_tensor,
        height,
        width,
        converted_from_colour,
    ) = prepare_image(dicom_path)

    if converted_from_colour:
        print(
            "Input note: this DICOM contains a three-channel colour image; "
            "it was converted to grayscale for the model."
        )

    # The checkpoint supplies every model weight, so no ImageNet download is
    # needed during inference.
    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=False,
    ).to(device)
    checkpoint = load_checkpoint(model, device)

    _, mask = predict_mask(
        model,
        image_tensor,
        device,
        output_size=(height, width),
    )

    mask_path, preview_path, figure = save_results(
        dicom_path,
        image,
        mask,
    )

    predicted_pixels = int(mask.sum())
    predicted_percentage = 100.0 * predicted_pixels / mask.size

    print("\n## Prediction complete")
    print(
        f"Original image size: {width} x {height} pixels"
    )
    print(f"Predicted mask pixels: {predicted_pixels:,}")
    print(
        "Predicted image area: "
        f"{predicted_percentage:.2f}%"
    )
    print(f"Binary mask: {mask_path.resolve()}")
    print(f"Preview image: {preview_path.resolve()}")

    if predicted_pixels == 0:
        print(
            "Result: the model did not suggest a pneumothorax region "
            "at the selected threshold."
        )
    else:
        print(
            "Result: the highlighted region is an initial suggestion "
            "and must be reviewed by a human annotator."
        )

    print("\nClose the preview window to finish the script.")
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()