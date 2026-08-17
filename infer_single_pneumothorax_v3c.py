"""Memory-conscious DICOM/PNG inference for the locked V3C model.

This inference-only script accepts one DICOM chest X-ray or one PNG image. It
loads a slim deployment checkpoint containing model weights and identity
metadata only. CPU libraries are limited to one worker thread, checkpoint
weights are memory-mapped, and the preview is generated with Pillow rather
than Matplotlib.

The model remains locked to epoch 5, 512 x 512 input and threshold 0.35.
Human review of every suggested mask is required.
"""

import os

# These must be set before NumPy, PyTorch or another numerical library is
# imported. They prevent OpenBLAS/OpenMP from allocating many CPU workers on a
# low-memory computer.
for variable_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable_name] = "1"

import argparse
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_PATH = (
    Path("checkpoints")
    / "pneumothorax_512_v3c_epoch5_deployment.pth"
)
OUTPUT_DIRECTORY = Path("inference_outputs_v3c")

EXPECTED_DEPLOYMENT_FORMAT = "inference_only_model_state_v1"
EXPECTED_TRAINING_STAGE = (
    "pneumothorax_512_v3c_batchnorm_stabilized_finetune"
)
EXPECTED_COMPLETED_EPOCH = 5
EXPECTED_IMAGE_SIZE = 512
EXPECTED_THRESHOLD = 0.35
EXPECTED_BEST_SCORE = 0.5070380715753282
EXPECTED_CONTROLLED_CHANGE = (
    "freeze BatchNorm running statistics during V3B training"
)
EXPECTED_BATCHNORM_MODE = (
    "saved_running_statistics_during_training"
)
SUPPORTED_SUFFIXES = {".dcm", ".png"}


class BasicResidualBlock(nn.Module):
    """Torchvision-compatible ResNet BasicBlock without torchvision imports."""

    def __init__(
        self,
        input_channels,
        output_channels,
        stride=1,
        downsample=None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        identity = inputs

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        outputs += identity
        return self.relu(outputs)


class DoubleConvolution(nn.Module):
    """Two decoder convolutions matching the trained model exactly."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class DecoderBlock(nn.Module):
    """Upsample and combine one decoder level with its skip features."""

    def __init__(
        self,
        input_channels,
        skip_channels,
        output_channels,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=2,
            stride=2,
        )
        self.convolutions = DoubleConvolution(
            output_channels + skip_channels,
            output_channels,
        )

    def forward(self, inputs, skip_features=None):
        outputs = self.upsample(inputs)

        if skip_features is not None:
            if outputs.shape[-2:] != skip_features.shape[-2:]:
                raise ValueError(
                    "Decoder and skip-connection dimensions do not match."
                )
            outputs = torch.cat((outputs, skip_features), dim=1)

        return self.convolutions(outputs)


class PneumothoraxDeploymentModel(nn.Module):
    """Exact ResNet34-U-Net topology without importing torchvision.

    Attribute and submodule names intentionally match the training model so
    strict state-dict loading verifies architectural identity.
    """

    def __init__(self):
        super().__init__()
        self._current_channels = 64

        self.encoder_stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder_pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.encoder_1 = self._make_residual_layer(
            output_channels=64,
            block_count=3,
            stride=1,
        )
        self.encoder_2 = self._make_residual_layer(
            output_channels=128,
            block_count=4,
            stride=2,
        )
        self.encoder_3 = self._make_residual_layer(
            output_channels=256,
            block_count=6,
            stride=2,
        )
        self.encoder_4 = self._make_residual_layer(
            output_channels=512,
            block_count=3,
            stride=2,
        )

        self.decoder_4 = DecoderBlock(512, 256, 256)
        self.decoder_3 = DecoderBlock(256, 128, 128)
        self.decoder_2 = DecoderBlock(128, 64, 64)
        self.decoder_1 = DecoderBlock(64, 64, 64)
        self.decoder_0 = DecoderBlock(64, 0, 32)
        self.output_layer = nn.Conv2d(32, 1, kernel_size=1)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_standard_deviation",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

    def _make_residual_layer(
        self,
        output_channels,
        block_count,
        stride,
    ):
        downsample = None
        if stride != 1 or self._current_channels != output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self._current_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

        blocks = [
            BasicResidualBlock(
                self._current_channels,
                output_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self._current_channels = output_channels

        for _ in range(1, block_count):
            blocks.append(
                BasicResidualBlock(
                    self._current_channels,
                    output_channels,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(
                "Expected grayscale inputs shaped "
                "(batch, 1, height, width)."
            )
        if images.shape[-2] % 32 or images.shape[-1] % 32:
            raise ValueError(
                "Image height and width must be divisible by 32."
            )

        rgb_images = images.repeat(1, 3, 1, 1)
        normalized = (
            rgb_images - self.imagenet_mean
        ) / self.imagenet_standard_deviation

        stem_features = self.encoder_stem(normalized)
        encoder_1 = self.encoder_1(
            self.encoder_pool(stem_features)
        )
        encoder_2 = self.encoder_2(encoder_1)
        encoder_3 = self.encoder_3(encoder_2)
        encoder_4 = self.encoder_4(encoder_3)

        decoder_4 = self.decoder_4(encoder_4, encoder_3)
        decoder_3 = self.decoder_3(decoder_4, encoder_2)
        decoder_2 = self.decoder_2(decoder_3, encoder_1)
        decoder_1 = self.decoder_1(decoder_2, stem_features)
        decoder_0 = self.decoder_0(decoder_1)
        return self.output_layer(decoder_0)


def parse_arguments():
    """Read an optional DICOM/PNG path and preview preference."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a V3C pneumothorax mask for one DICOM or PNG "
            "chest X-ray. A file-selection window opens when no path "
            "is supplied."
        )
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        help="Optional path to one .dcm or .png image.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Save the preview without opening the system image viewer.",
    )
    return parser.parse_args()


def choose_image_file(command_line_path):
    """Use the supplied path or open a DICOM/PNG file picker."""

    if command_line_path is not None:
        path = command_line_path.expanduser()
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as error:
            raise RuntimeError(
                "The file picker is unavailable. Supply a path, for "
                "example: python infer_single_pneumothorax_v3c.py image.png"
            ) from error

        root = tk.Tk()
        root.withdraw()
        root.update()
        selected_path = filedialog.askopenfilename(
            title="Choose one DICOM or PNG chest X-ray",
            filetypes=(
                ("Supported images", "*.dcm *.png"),
                ("DICOM files", "*.dcm"),
                ("PNG files", "*.png"),
                ("All files", "*.*"),
            ),
        )
        root.destroy()

        if not selected_path:
            raise SystemExit("No image was selected.")

        path = Path(selected_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"The selected image was not found: {path.resolve()}"
        )

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported image type {suffix!r}. Use a .dcm or .png file."
        )

    return path


def choose_device():
    """Use CUDA when available; otherwise use memory-limited CPU inference."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_torch_cpu():
    """Prevent PyTorch from creating unnecessary CPU worker pools."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this setting only before inter-op work starts.
        pass


def values_match(actual, expected):
    """Compare numeric checkpoint metadata strictly."""

    try:
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    except (TypeError, ValueError):
        return False


def validate_checkpoint_metadata(checkpoint):
    """Reject any checkpoint other than the selected slim V3C export."""

    if checkpoint.get("deployment_format") != EXPECTED_DEPLOYMENT_FORMAT:
        raise ValueError(
            "This is not the required slim V3C deployment checkpoint. "
            "Create it using the supplied Colab export cell."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError("The deployment checkpoint has no model_state_dict.")

    forbidden_training_state = {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
    }
    unexpected = forbidden_training_state.intersection(checkpoint)
    if unexpected:
        raise ValueError(
            "The deployment checkpoint unexpectedly contains training "
            f"state: {sorted(unexpected)}."
        )

    direct_checks = {
        "training stage": (
            checkpoint.get("training_stage"),
            EXPECTED_TRAINING_STAGE,
        ),
        "completed epoch": (
            checkpoint.get("completed_epoch"),
            EXPECTED_COMPLETED_EPOCH,
        ),
    }
    for description, (actual, expected) in direct_checks.items():
        if actual != expected:
            raise ValueError(
                f"Unexpected V3C {description}: {actual!r}; "
                f"expected {expected!r}."
            )

    configuration = checkpoint.get("configuration", {})
    configuration_checks = {
        "image size": (
            configuration.get("image_size"),
            EXPECTED_IMAGE_SIZE,
        ),
        "controlled change": (
            configuration.get("controlled_change"),
            EXPECTED_CONTROLLED_CHANGE,
        ),
        "BatchNorm mode": (
            configuration.get("batchnorm_mode"),
            EXPECTED_BATCHNORM_MODE,
        ),
        "BatchNorm running-statistic freeze": (
            configuration.get(
                "batchnorm_running_statistics_frozen"
            ),
            True,
        ),
        "validation split confirmation": (
            configuration.get("validation_split_used"),
            True,
        ),
        "test split confirmation": (
            configuration.get("test_split_used"),
            False,
        ),
    }
    for description, (actual, expected) in configuration_checks.items():
        if actual != expected:
            raise ValueError(
                f"Unexpected V3C {description}: {actual!r}; "
                f"expected {expected!r}."
            )

    threshold = configuration.get("prediction_threshold")
    if not values_match(threshold, EXPECTED_THRESHOLD):
        raise ValueError(
            f"Unexpected threshold {threshold!r}; "
            f"expected {EXPECTED_THRESHOLD}."
        )

    best_score = checkpoint.get("best_validation_selection_score")
    if not values_match(best_score, EXPECTED_BEST_SCORE):
        raise ValueError(
            f"Unexpected validation score {best_score!r}; "
            f"expected {EXPECTED_BEST_SCORE}."
        )


def load_locked_model(device):
    """Memory-map the slim weights and assign them to a meta-device model."""

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "The slim V3C deployment checkpoint was not found at: "
            f"{CHECKPOINT_PATH.resolve()}"
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "This memory-conscious loader requires a recent PyTorch version "
            "with weights_only and mmap support."
        ) from error

    validate_checkpoint_metadata(checkpoint)

    # A meta-device model allocates no random parameter storage. assign=True
    # then attaches the memory-mapped checkpoint tensors without making a
    # second complete CPU copy of the model.
    with torch.device("meta"):
        model = PneumothoraxDeploymentModel()

    try:
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
            assign=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "This memory-conscious loader requires load_state_dict(assign=True)."
        ) from error

    model = model.to(device)
    model.eval()
    return model, checkpoint


def convert_to_grayscale(image):
    """Return a contiguous two-dimensional grayscale image."""

    if image.ndim == 2:
        return np.ascontiguousarray(image), False

    if image.ndim != 3:
        raise ValueError(
            f"Expected a 2D grayscale or RGB image; found shape {image.shape}."
        )

    if image.shape[-1] == 2:
        # PNG luminance-alpha: retain luminance and ignore alpha.
        return np.ascontiguousarray(image[..., 0]), True
    if image.shape[-1] in (3, 4):
        colour = image[..., :3]
    elif image.shape[0] in (3, 4):
        colour = np.moveaxis(image[:3], 0, -1)
    else:
        raise ValueError(
            f"Unsupported colour-channel layout: {image.shape}."
        )

    grayscale = np.sum(
        colour.astype(np.float32)
        * np.array([0.299, 0.587, 0.114], dtype=np.float32),
        axis=-1,
    )
    return np.ascontiguousarray(grayscale), True


def min_max_normalise(image):
    """Match the per-image normalization used during SIIM training."""

    image = image.astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("The input image contains non-finite pixel values.")

    minimum = float(image.min())
    maximum = float(image.max())
    if maximum <= minimum:
        raise ValueError("The input image has no usable intensity range.")

    return ((image - minimum) / (maximum - minimum)).astype(np.float32)


def load_dicom(path):
    """Load, orient and normalize one DICOM chest X-ray."""

    # Imported lazily so PNG-only deployment does not allocate DICOM modules.
    import pydicom

    dicom_data = pydicom.dcmread(path)
    image = dicom_data.pixel_array.astype(np.float32)

    slope = float(getattr(dicom_data, "RescaleSlope", 1.0))
    intercept = float(getattr(dicom_data, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    photometric = str(
        getattr(dicom_data, "PhotometricInterpretation", "")
    ).upper()
    if photometric == "MONOCHROME1":
        image = image.max() + image.min() - image

    image, converted = convert_to_grayscale(image)
    return min_max_normalise(image), converted


def load_png(path):
    """Load and normalize one PNG without changing its spatial orientation."""

    with Image.open(path) as pil_image:
        if pil_image.mode == "P":
            pil_image = pil_image.convert("RGB")
        image = np.asarray(pil_image)

    image, converted = convert_to_grayscale(image)
    return min_max_normalise(image), converted


def prepare_image(path):
    """Load one supported image and create the 512 x 512 model tensor."""

    if path.suffix.lower() == ".dcm":
        image, converted = load_dicom(path)
        source_type = "DICOM"
    else:
        image, converted = load_png(path)
        source_type = "PNG"

    height, width = image.shape
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return image, tensor, height, width, source_type, converted


@torch.inference_mode()
def predict_mask(model, image_tensor, device, output_size):
    """Predict at 512 x 512 and restore the original image dimensions."""

    probabilities = torch.sigmoid(model(image_tensor.to(device)))
    expected_shape = (
        1,
        1,
        EXPECTED_IMAGE_SIZE,
        EXPECTED_IMAGE_SIZE,
    )
    if tuple(probabilities.shape) != expected_shape:
        raise ValueError(
            f"Unexpected model output shape: {tuple(probabilities.shape)}"
        )

    mask = probabilities >= EXPECTED_THRESHOLD
    mask = F.interpolate(
        mask.float(),
        size=output_size,
        mode="nearest",
    )
    return mask.squeeze().cpu().numpy().astype(bool)


def safe_output_stem(path):
    """Create a filesystem-safe output prefix."""

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return stem or "image_prediction"


def preview_panel(image):
    """Convert a normalized image into a compact RGB preview panel."""

    height, width = image.shape
    scale = min(1.0, 768.0 / max(height, width))
    preview_width = max(1, round(width * scale))
    preview_height = max(1, round(height * scale))

    grayscale = Image.fromarray(
        np.round(image * 255.0).astype(np.uint8)
    )
    grayscale = grayscale.resize(
        (preview_width, preview_height),
        resample=Image.Resampling.BILINEAR,
    )
    return grayscale.convert("RGB")


def save_results(path, image, mask):
    """Save a full-resolution binary mask and a compact three-panel preview."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_stem = safe_output_stem(path)
    mask_path = OUTPUT_DIRECTORY / f"{output_stem}_mask.png"
    preview_path = OUTPUT_DIRECTORY / f"{output_stem}_preview.png"

    full_mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    full_mask_image.save(mask_path)

    image_panel = preview_panel(image)
    panel_size = image_panel.size
    mask_panel = full_mask_image.resize(
        panel_size,
        resample=Image.Resampling.NEAREST,
    ).convert("RGB")

    image_array = np.asarray(image_panel).copy()
    preview_mask = np.asarray(mask_panel.convert("L")) > 0
    overlay_array = image_array.copy()
    if preview_mask.any():
        red = np.array([255, 0, 0], dtype=np.float32)
        overlay_array[preview_mask] = np.round(
            0.55 * overlay_array[preview_mask].astype(np.float32)
            + 0.45 * red
        ).astype(np.uint8)
    overlay_panel = Image.fromarray(overlay_array)

    title_height = 36
    gap = 8
    canvas = Image.new(
        "RGB",
        (
            panel_size[0] * 3 + gap * 2,
            panel_size[1] + title_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    titles = (
        "Input image",
        f"Mask (threshold {EXPECTED_THRESHOLD:.2f})",
        "Suggested overlay",
    )
    panels = (image_panel, mask_panel, overlay_panel)
    for index, (title, panel) in enumerate(zip(titles, panels)):
        left = index * (panel_size[0] + gap)
        draw.text((left + 8, 10), title, fill="black")
        canvas.paste(panel, (left, title_height))

    canvas.save(preview_path)
    return mask_path, preview_path, canvas


def main():
    args = parse_arguments()
    configure_torch_cpu()
    image_path = choose_image_file(args.image_path)
    device = choose_device()

    print("Locked V3C epoch-5 DICOM/PNG inference")
    print("--------------------------------------")
    print(f"Device: {device}")
    print(f"Selected image: {image_path.resolve()}")
    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print(f"Model input size: {EXPECTED_IMAGE_SIZE} x {EXPECTED_IMAGE_SIZE}")
    print(f"Prediction threshold: {EXPECTED_THRESHOLD}")
    print("Human review required; this is not an autonomous diagnosis.")

    (
        image,
        image_tensor,
        height,
        width,
        source_type,
        converted,
    ) = prepare_image(image_path)
    print(f"Input type: {source_type}")
    if converted:
        print("Input note: colour/alpha channels were converted to grayscale.")
    if source_type == "PNG":
        print(
            "PNG note: the image must use the normal chest-X-ray display "
            "orientation (bright bones, dark lungs)."
        )

    model, checkpoint = load_locked_model(device)
    print(f"Checkpoint epoch: {checkpoint['completed_epoch']}")
    print(f"Training stage: {checkpoint['training_stage']}")
    print("Checkpoint format: slim memory-mapped deployment weights")

    mask = predict_mask(
        model,
        image_tensor,
        device,
        output_size=(height, width),
    )
    mask_path, preview_path, preview = save_results(
        image_path,
        image,
        mask,
    )

    predicted_pixels = int(mask.sum())
    predicted_percentage = 100.0 * predicted_pixels / mask.size

    print("\nPrediction complete")
    print("-------------------")
    print(f"Original image size: {width} x {height} pixels")
    print(f"Predicted mask pixels: {predicted_pixels:,}")
    print(f"Predicted image area: {predicted_percentage:.2f}%")
    print(f"Binary mask: {mask_path.resolve()}")
    print(f"Preview image: {preview_path.resolve()}")

    if predicted_pixels == 0:
        print(
            "Result: no pneumothorax region was suggested at the locked "
            "threshold."
        )
    else:
        print(
            "Result: the highlighted area is an initial suggestion and "
            "must be reviewed by a human annotator."
        )

    if not args.no_open:
        preview.show(
            title="V3C pneumothorax suggestion — human review required"
        )


if __name__ == "__main__":
    main()
