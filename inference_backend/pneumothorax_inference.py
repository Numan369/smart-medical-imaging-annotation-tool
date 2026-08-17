\
"""Backend-ready inference for the frozen pneumothorax model.

Research prototype only. Every generated mask requires human review
and must not be interpreted as a medical diagnosis.
"""

from contextlib import nullcontext
from pathlib import Path
import argparse
import hashlib
import json
import re
import time

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from pneumothorax_model import PneumothoraxResNet34UNet
from v4a_frozen_postprocessing import (
    IMAGE_SIZE,
    MINIMUM_COMPONENT_PIXELS,
    PROBABILITY_THRESHOLD,
    apply_v4a_postprocessing,
)


EXPECTED_CHECKPOINT_SHA256 = "109e102e7a521abc6c904e1c5ad214e1a3f18a5b3bfd3dc0d51c98be4a585635"
EXPECTED_CHECKPOINT_EPOCH = 10
EXPECTED_TRAINING_STAGE = (
    "pneumothorax_512_negative_aware_finetune"
)

DISCLAIMER = (
    "AI-generated suggestions require human review and must not "
    "be treated as a medical diagnosis."
)

SUPPORTED_RASTER_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}

SUPPORTED_DICOM_EXTENSIONS = {
    ".dcm",
    ".dicom",
}


def sha256_file(path):
    """Return a streaming SHA-256 hash for a file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as source:
        while True:
            block = source.read(8 * 1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def normalise_array(image):
    """Min-max normalize a two-dimensional image to [0, 1]."""

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim != 2:
        raise ValueError(
            "Expected one two-dimensional grayscale image."
        )

    if not np.isfinite(image).all():
        raise ValueError(
            "The image contains a non-finite intensity value."
        )

    minimum = float(image.min())
    maximum = float(image.max())

    if maximum <= minimum:
        raise ValueError(
            "The image contains no usable intensity range."
        )

    image = (
        (image - minimum)
        / (maximum - minimum)
    )

    return image.astype(np.float32)


def load_dicom_image(path):
    """Load a DICOM using the same normalization as training."""

    dicom_data = pydicom.dcmread(path)

    image = dicom_data.pixel_array.astype(
        np.float32
    )

    if image.ndim != 2:
        raise ValueError(
            "Only single-frame two-dimensional DICOM X-rays "
            "are currently supported."
        )

    slope = float(
        getattr(dicom_data, "RescaleSlope", 1.0)
    )

    intercept = float(
        getattr(dicom_data, "RescaleIntercept", 0.0)
    )

    image = image * slope + intercept

    photometric = str(
        getattr(
            dicom_data,
            "PhotometricInterpretation",
            "",
        )
    ).upper()

    if photometric == "MONOCHROME1":
        image = image.max() + image.min() - image

    image = normalise_array(image)

    return image, "dicom"


def load_raster_image(path):
    """Load PNG/JPG and convert it into one normalized channel."""

    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)

        if source.mode in {
            "I",
            "I;16",
            "I;16B",
            "I;16L",
            "F",
        }:
            image = np.asarray(
                source,
                dtype=np.float32,
            )
        else:
            grayscale = source.convert("L")

            image = np.asarray(
                grayscale,
                dtype=np.float32,
            )

    image = normalise_array(image)

    return image, "raster"


def load_medical_image(path):
    """Load one supported DICOM or raster image."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix in SUPPORTED_DICOM_EXTENSIONS:
        return load_dicom_image(path)

    if suffix in SUPPORTED_RASTER_EXTENSIONS:
        return load_raster_image(path)

    raise ValueError(
        "Unsupported image type. Supported extensions are: "
        ".dcm, .dicom, .png, .jpg and .jpeg."
    )


def resize_for_model(image):
    """Resize normalized grayscale input exactly as during training."""

    image_tensor = (
        torch.from_numpy(image)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    image_tensor = F.interpolate(
        image_tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return image_tensor


def resize_mask_to_original(mask, original_shape):
    """Return the filtered model mask at the input image size."""

    mask_tensor = (
        torch.from_numpy(
            mask.astype(np.float32)
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    resized = F.interpolate(
        mask_tensor,
        size=original_shape,
        mode="nearest",
    )

    return (
        resized
        .squeeze(0)
        .squeeze(0)
        .numpy()
        .astype(np.uint8)
    )


class PneumothoraxInferenceEngine:
    """Load the frozen model once and process individual images."""

    def __init__(
        self,
        checkpoint_path,
        device=None,
        verify_checkpoint_hash=True,
    ):
        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                self.checkpoint_path
            )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        if verify_checkpoint_hash:
            observed_hash = sha256_file(
                self.checkpoint_path
            )

            if observed_hash != EXPECTED_CHECKPOINT_SHA256:
                raise ValueError(
                    "The checkpoint does not match the frozen "
                    "epoch-10 model."
                )

        # No ImageNet download is required because the full trained
        # model state is loaded immediately afterward.
        self.model = PneumothoraxResNet34UNet(
            use_pretrained_encoder=False,
            freeze_encoder=True,
        )

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
            )

        if (
            checkpoint.get("completed_epoch")
            != EXPECTED_CHECKPOINT_EPOCH
        ):
            raise ValueError(
                "Expected the epoch-10 best checkpoint."
            )

        if (
            checkpoint.get("training_stage")
            != EXPECTED_TRAINING_STAGE
        ):
            raise ValueError(
                "The checkpoint training stage is incorrect."
            )

        load_result = self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        if (
            load_result.missing_keys
            or load_result.unexpected_keys
        ):
            raise ValueError(
                "Checkpoint state does not exactly match "
                "the model architecture."
            )

        self.model = self.model.to(
            self.device
        )
        self.model.eval()


    def predict(self, image_path):
        """Run inference and return masks plus frontend-ready values."""

        total_start = time.perf_counter()

        normalized_image, input_format = (
            load_medical_image(image_path)
        )

        original_shape = tuple(
            int(value)
            for value in normalized_image.shape
        )

        model_input = resize_for_model(
            normalized_image
        ).to(
            self.device,
            non_blocking=self.device.type == "cuda",
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        model_start = time.perf_counter()

        with torch.inference_mode():
            if self.device.type == "cuda":
                autocast_context = torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                )
            else:
                autocast_context = nullcontext()

            with autocast_context:
                logits = self.model(model_input)

            probability_map = (
                torch.sigmoid(logits)
                .squeeze(0)
                .squeeze(0)
                .float()
                .cpu()
                .numpy()
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        model_seconds = (
            time.perf_counter() - model_start
        )

        model_mask, region_count = (
            apply_v4a_postprocessing(
                probability_map
            )
        )

        original_mask = resize_mask_to_original(
            model_mask,
            original_shape,
        )

        mask_pixels = int(model_mask.sum())

        mask_coverage_percent = (
            100.0
            * mask_pixels
            / (IMAGE_SIZE * IMAGE_SIZE)
        )

        maximum_output_score = float(
            probability_map.max()
        )

        if mask_pixels > 0:
            finding = "possible-region-detected"
            finding_title = (
                "Possible pneumothorax region detected"
            )
            finding_explanation = (
                "AI identified one or more regions that may "
                "represent pneumothorax. Review the segmentation "
                "before accepting it."
            )
        else:
            finding = "no-region-detected"
            finding_title = (
                "No pneumothorax region detected by AI"
            )
            finding_explanation = (
                "The AI did not generate a pneumothorax region "
                "at the selected operating threshold. Human "
                "review is still required."
            )

        processing_time_ms = (
            1000.0
            * (time.perf_counter() - total_start)
        )

        summary = {
            "finding": finding,
            "findingTitle": finding_title,
            "findingExplanation":
                finding_explanation,
            "regionCount": int(region_count),
            "maskCoveragePercent":
                mask_coverage_percent,
            "maximumOutputScore":
                maximum_output_score,
            "threshold":
                PROBABILITY_THRESHOLD,
            "minimumComponentPixels":
                MINIMUM_COMPONENT_PIXELS,
            "processingTimeMs":
                processing_time_ms,
            "modelInferenceTimeMs":
                1000.0 * model_seconds,
            "reviewStatus":
                "awaiting-review",
            "inputFormat": input_format,
            "originalImageShape": list(
                original_shape
            ),
            "modelInputShape": [
                IMAGE_SIZE,
                IMAGE_SIZE,
            ],
            "referenceMetrics": None,
            "warning": DISCLAIMER,
        }

        return {
            "summary": summary,
            "normalized_image":
                normalized_image,
            "probability_map":
                probability_map,
            "model_mask":
                model_mask.astype(np.uint8),
            "original_size_mask":
                original_mask,
        }


def safe_stem(path):
    """Create a filesystem-safe output name."""

    stem = Path(path).stem

    safe = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        stem,
    ).strip("._")

    return safe or "medical_image"


def save_prediction(
    prediction,
    image_path,
    output_directory,
):
    """Save preview, mask, overlay and JSON summary."""

    output_directory = Path(
        output_directory
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = safe_stem(image_path)

    preview_path = (
        output_directory
        / f"{stem}_normalized_preview.png"
    )

    mask_path = (
        output_directory
        / f"{stem}_ai_mask.png"
    )

    overlay_path = (
        output_directory
        / f"{stem}_ai_overlay.png"
    )

    summary_path = (
        output_directory
        / f"{stem}_ai_result.json"
    )

    normalized_image = prediction[
        "normalized_image"
    ]

    original_mask = prediction[
        "original_size_mask"
    ].astype(bool)

    preview_uint8 = np.clip(
        normalized_image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(
        preview_uint8,
        mode="L",
    ).save(preview_path)

    Image.fromarray(
        original_mask.astype(np.uint8) * 255,
        mode="L",
    ).save(mask_path)

    overlay = np.stack(
        [preview_uint8] * 3,
        axis=-1,
    ).astype(np.float32)

    cyan = np.array(
        [6, 182, 212],
        dtype=np.float32,
    )

    overlay[original_mask] = (
        0.52 * overlay[original_mask]
        + 0.48 * cyan
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(
        overlay,
        mode="RGB",
    ).save(overlay_path)

    summary_path.write_text(
        json.dumps(
            prediction["summary"],
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "preview": str(preview_path),
        "mask": str(mask_path),
        "overlay": str(overlay_path),
        "summary": str(summary_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a human-review-required "
            "pneumothorax mask suggestion."
        )
    )

    parser.add_argument(
        "image",
        type=Path,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
    )

    arguments = parser.parse_args()

    engine = PneumothoraxInferenceEngine(
        checkpoint_path=arguments.checkpoint,
        device=arguments.device,
    )

    prediction = engine.predict(
        arguments.image
    )

    paths = save_prediction(
        prediction,
        arguments.image,
        arguments.output_directory,
    )

    print(
        json.dumps(
            {
                "result":
                    prediction["summary"],
                "savedFiles": paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
