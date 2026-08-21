from pathlib import Path
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

from tn3k_model import TN3KResNet34UNet


# ============================================================
# PATHS
# ============================================================

THYROID_ROOT = (
    Path(__file__).resolve().parent.parent
)


DEFAULT_CHECKPOINT = (
    THYROID_ROOT
    / "checkpoints"
    / "tn3k_v1_earlystop"
    / "tn3k_v1_earlystop_best.pth"
)


DEFAULT_OUTPUT_DIR = (
    THYROID_ROOT
    / "outputs"
    / "tn3k_v1_inference"
)


DEFAULT_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOCKED FINAL MODEL CONFIGURATION
# ============================================================

IMAGE_SIZE = 512

PREDICTION_THRESHOLD = 0.50

EXPECTED_CHECKPOINT_EPOCH = 15

EXPECTED_VALIDATION_DICE = 0.808592


# ============================================================
# DEVICE
# ============================================================

def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(
    image_path,
):
    """
    Load an arbitrary ultrasound image and transform it into
    exactly the same 512x512 geometry used during training.

    Returns:
        model_tensor
        metadata
        original_image
    """

    image_path = Path(
        image_path
    )


    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )


    # --------------------------------------------------------
    # Load as grayscale.
    # --------------------------------------------------------

    pil_image = Image.open(
        image_path
    ).convert(
        "L"
    )


    original_width = (
        pil_image.width
    )

    original_height = (
        pil_image.height
    )


    original_image = np.array(
        pil_image,
        dtype=np.uint8,
    )


    # ========================================================
    # ASPECT-RATIO-PRESERVING RESIZE
    # ========================================================

    scale = min(

        IMAGE_SIZE
        / original_width,

        IMAGE_SIZE
        / original_height,
    )


    resized_width = int(
        round(
            original_width
            * scale
        )
    )


    resized_height = int(
        round(
            original_height
            * scale
        )
    )


    # Safety.
    resized_width = min(
        resized_width,
        IMAGE_SIZE,
    )


    resized_height = min(
        resized_height,
        IMAGE_SIZE,
    )


    resized_pil = pil_image.resize(

        (
            resized_width,
            resized_height,
        ),

        resample=Image.Resampling.BILINEAR,
    )


    resized_array = np.array(
        resized_pil,
        dtype=np.float32,
    )


    resized_array = (
        resized_array
        / 255.0
    )


    # ========================================================
    # CENTER PAD TO 512x512
    # ========================================================

    canvas = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        dtype=np.float32,
    )


    pad_left = (
        IMAGE_SIZE
        - resized_width
    ) // 2


    pad_top = (
        IMAGE_SIZE
        - resized_height
    ) // 2


    pad_right = (
        IMAGE_SIZE
        - resized_width
        - pad_left
    )


    pad_bottom = (
        IMAGE_SIZE
        - resized_height
        - pad_top
    )


    canvas[
        pad_top:
        pad_top + resized_height,

        pad_left:
        pad_left + resized_width,
    ] = resized_array


    # --------------------------------------------------------
    # [H,W]
    # ->
    # [1,1,H,W]
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        canvas
    ).float()


    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )


    metadata = {

        "original_width":
            original_width,

        "original_height":
            original_height,

        "resized_width":
            resized_width,

        "resized_height":
            resized_height,

        "pad_left":
            pad_left,

        "pad_right":
            pad_right,

        "pad_top":
            pad_top,

        "pad_bottom":
            pad_bottom,

        "scale":
            scale,
    }


    return (
        tensor,
        metadata,
        original_image,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    checkpoint_path = Path(
        checkpoint_path
    )


    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{checkpoint_path}"
        )


    print()
    print(
        "Loading checkpoint:"
    )

    print(
        checkpoint_path
    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=device,

        weights_only=True,
    )


    checkpoint_epoch = int(
        checkpoint.get(
            "epoch",
            -1,
        )
    )


    validation_dice = float(
        checkpoint.get(
            "best_validation_dice",
            -1.0,
        )
    )


    # ========================================================
    # VERIFY LOCKED MODEL
    # ========================================================

    if (
        checkpoint_epoch
        != EXPECTED_CHECKPOINT_EPOCH
    ):

        raise RuntimeError(
            "Unexpected checkpoint epoch.\n"
            f"Expected: "
            f"{EXPECTED_CHECKPOINT_EPOCH}\n"
            f"Found: "
            f"{checkpoint_epoch}"
        )


    if abs(
        validation_dice
        - EXPECTED_VALIDATION_DICE
    ) > 1e-5:

        raise RuntimeError(
            "Unexpected checkpoint validation Dice.\n"
            f"Expected: "
            f"{EXPECTED_VALIDATION_DICE:.6f}\n"
            f"Found: "
            f"{validation_dice:.6f}"
        )


    model = TN3KResNet34UNet(
        use_pretrained_encoder=False,
    )


    model.configure_v1_trainable_layers()


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model = model.to(
        device
    )


    model.eval()


    print()
    print(
        f"Checkpoint epoch: "
        f"{checkpoint_epoch}"
    )


    print(
        f"Validation Dice: "
        f"{validation_dice:.6f}"
    )


    print(
        f"Prediction threshold: "
        f"{PREDICTION_THRESHOLD:.2f}"
    )


    return (
        model,
        checkpoint,
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

@torch.no_grad()
def predict_model_space(
    model,
    tensor,
    device,
):

    tensor = tensor.to(
        device
    )


    logits = model(
        tensor
    )


    probability = torch.sigmoid(
        logits
    )


    probability = (
        probability[
            0,
            0
        ]
        .cpu()
        .numpy()
    )


    prediction = (
        probability
        >= PREDICTION_THRESHOLD
    ).astype(
        np.uint8
    )


    return (
        probability,
        prediction,
    )


# ============================================================
# CONVERT MODEL MASK BACK TO ORIGINAL IMAGE SPACE
# ============================================================

def restore_mask_to_original_size(
    model_prediction,
    metadata,
):
    """
    Remove the 512x512 padding and resize the binary mask back
    to the exact width and height of the uploaded ultrasound.
    """

    pad_top = metadata[
        "pad_top"
    ]


    pad_left = metadata[
        "pad_left"
    ]


    resized_height = metadata[
        "resized_height"
    ]


    resized_width = metadata[
        "resized_width"
    ]


    original_width = metadata[
        "original_width"
    ]


    original_height = metadata[
        "original_height"
    ]


    # ========================================================
    # REMOVE CENTER PADDING
    # ========================================================

    cropped_mask = model_prediction[

        pad_top:
        pad_top + resized_height,

        pad_left:
        pad_left + resized_width,
    ]


    # ========================================================
    # RESIZE TO ORIGINAL IMAGE DIMENSIONS
    #
    # Nearest-neighbour preserves binary segmentation labels.
    # ========================================================

    mask_pil = Image.fromarray(
        (
            cropped_mask
            * 255
        ).astype(
            np.uint8
        )
    )


    restored_pil = mask_pil.resize(

        (
            original_width,
            original_height,
        ),

        resample=Image.Resampling.NEAREST,
    )


    restored_mask = (

        np.array(
            restored_pil
        )
        >= 128

    ).astype(
        np.uint8
    )


    return restored_mask


# ============================================================
# RESTORE PROBABILITY MAP
# ============================================================

def restore_probability_to_original_size(
    probability,
    metadata,
):

    pad_top = metadata[
        "pad_top"
    ]


    pad_left = metadata[
        "pad_left"
    ]


    resized_height = metadata[
        "resized_height"
    ]


    resized_width = metadata[
        "resized_width"
    ]


    original_width = metadata[
        "original_width"
    ]


    original_height = metadata[
        "original_height"
    ]


    cropped_probability = probability[

        pad_top:
        pad_top + resized_height,

        pad_left:
        pad_left + resized_width,
    ]


    # Convert probability [0,1] to 16-bit temporarily so
    # interpolation does not destroy most of its precision.
    probability_uint16 = (

        np.clip(
            cropped_probability,
            0.0,
            1.0,
        )

        * 65535.0

    ).astype(
        np.uint16
    )


    probability_pil = Image.fromarray(
        probability_uint16
    )


    restored_pil = probability_pil.resize(

        (
            original_width,
            original_height,
        ),

        resample=Image.Resampling.BILINEAR,
    )


    restored_probability = (

        np.array(
            restored_pil,
            dtype=np.float32,
        )

        / 65535.0
    )


    return restored_probability


# ============================================================
# SAVE MASK
# ============================================================

def save_binary_mask(
    mask,
    output_path,
):

    output = (

        mask
        * 255

    ).astype(
        np.uint8
    )


    Image.fromarray(
        output
    ).save(
        output_path
    )


# ============================================================
# SAVE PROBABILITY MAP
# ============================================================

def save_probability_map(
    probability,
    output_path,
):

    output = (

        np.clip(
            probability,
            0.0,
            1.0,
        )

        * 255.0

    ).astype(
        np.uint8
    )


    Image.fromarray(
        output
    ).save(
        output_path
    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    original_image,
    restored_probability,
    restored_mask,
    output_path,
    image_name,
):

    figure, axes = plt.subplots(

        1,
        4,

        figsize=(16, 4),
    )


    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    axes[0].imshow(
        original_image,
        cmap="gray",
    )


    axes[0].set_title(
        "Original Ultrasound"
    )


    axes[0].axis(
        "off"
    )


    # --------------------------------------------------------
    # Probability heatmap
    # --------------------------------------------------------

    probability_display = (
        axes[1].imshow(

            restored_probability,

            vmin=0.0,

            vmax=1.0,
        )
    )


    axes[1].set_title(
        "AI Probability"
    )


    axes[1].axis(
        "off"
    )


    figure.colorbar(

        probability_display,

        ax=axes[1],

        fraction=0.046,

        pad=0.04,
    )


    # --------------------------------------------------------
    # Binary mask
    # --------------------------------------------------------

    axes[2].imshow(

        restored_mask,

        cmap="gray",

        vmin=0,

        vmax=1,
    )


    axes[2].set_title(
        "AI Nodule Mask"
    )


    axes[2].axis(
        "off"
    )


    # --------------------------------------------------------
    # Original + AI contour
    # --------------------------------------------------------

    axes[3].imshow(
        original_image,
        cmap="gray",
    )


    if restored_mask.max() > 0:

        axes[3].contour(

            restored_mask,

            levels=[0.5],

            colors="red",

            linewidths=2,
        )


    axes[3].set_title(
        "AI Suggestion"
    )


    axes[3].axis(
        "off"
    )


    figure.suptitle(

        f"TN3K AI Nodule Segmentation\n"
        f"{image_name}"
    )


    plt.tight_layout()


    figure.savefig(

        output_path,

        dpi=160,

        bbox_inches="tight",
    )


    plt.close(
        figure
    )


# ============================================================
# COMPLETE SINGLE-IMAGE INFERENCE
# ============================================================

def infer_single_image(
    image_path,
    checkpoint_path=DEFAULT_CHECKPOINT,
    output_dir=DEFAULT_OUTPUT_DIR,
):

    image_path = Path(
        image_path
    )


    output_dir = Path(
        output_dir
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()


    print("=" * 70)
    print(
        "TN3K V1 FINAL INFERENCE"
    )
    print("=" * 70)


    print()
    print(
        "Device:",
        device
    )


    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # ========================================================
    # PREPROCESS
    # ========================================================

    (
        tensor,
        metadata,
        original_image,
    ) = preprocess_image(
        image_path
    )


    print()
    print(
        "Input image:",
        image_path
    )


    print(
        "Original size:",
        f"{metadata['original_width']}"
        f"×"
        f"{metadata['original_height']}"
    )


    print(
        "Resized size:",
        f"{metadata['resized_width']}"
        f"×"
        f"{metadata['resized_height']}"
    )


    print(
        "Padding:"
    )


    print(
        f"  left   = "
        f"{metadata['pad_left']}"
    )


    print(
        f"  right  = "
        f"{metadata['pad_right']}"
    )


    print(
        f"  top    = "
        f"{metadata['pad_top']}"
    )


    print(
        f"  bottom = "
        f"{metadata['pad_bottom']}"
    )


    # ========================================================
    # MODEL
    # ========================================================

    (
        model,
        checkpoint,
    ) = load_model(

        checkpoint_path,

        device,
    )


    # ========================================================
    # PREDICT
    # ========================================================

    (
        model_probability,
        model_mask,
    ) = predict_model_space(

        model,

        tensor,

        device,
    )


    # ========================================================
    # RESTORE TO ORIGINAL IMAGE GEOMETRY
    # ========================================================

    restored_mask = (
        restore_mask_to_original_size(

            model_mask,

            metadata,
        )
    )


    restored_probability = (
        restore_probability_to_original_size(

            model_probability,

            metadata,
        )
    )


    # ========================================================
    # SAFETY
    # ========================================================

    expected_shape = (

        metadata[
            "original_height"
        ],

        metadata[
            "original_width"
        ],
    )


    if (
        restored_mask.shape
        != expected_shape
    ):

        raise RuntimeError(
            "Restored mask size mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Found: "
            f"{restored_mask.shape}"
        )


    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    stem = image_path.stem


    mask_path = (

        output_dir

        / f"{stem}_tn3k_mask.png"
    )


    probability_path = (

        output_dir

        / f"{stem}_tn3k_probability.png"
    )


    visualization_path = (

        output_dir

        / f"{stem}_tn3k_visualization.png"
    )


    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    save_binary_mask(

        restored_mask,

        mask_path,
    )


    save_probability_map(

        restored_probability,

        probability_path,
    )


    save_visualization(

        original_image=original_image,

        restored_probability=(
            restored_probability
        ),

        restored_mask=(
            restored_mask
        ),

        output_path=(
            visualization_path
        ),

        image_name=(
            image_path.name
        ),
    )


    # ========================================================
    # BASIC OUTPUT STATISTICS
    # ========================================================

    predicted_pixels = int(
        restored_mask.sum()
    )


    total_pixels = int(
        restored_mask.size
    )


    predicted_fraction = (

        predicted_pixels

        / total_pixels
    )


    max_probability = float(
        restored_probability.max()
    )


    mean_probability = float(
        restored_probability.mean()
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "INFERENCE COMPLETE"
    )
    print("=" * 70)


    print()
    print(
        "Final mask size:",
        restored_mask.shape
    )


    print(
        "Predicted nodule pixels:",
        predicted_pixels
    )


    print(
        "Predicted image fraction:",
        f"{predicted_fraction * 100:.2f}%"
    )


    print(
        "Maximum probability:",
        f"{max_probability:.4f}"
    )


    print(
        "Mean probability:",
        f"{mean_probability:.4f}"
    )


    print()
    print(
        "Mask:"
    )

    print(
        mask_path
    )


    print()
    print(
        "Probability map:"
    )

    print(
        probability_path
    )


    print()
    print(
        "Visualization:"
    )

    print(
        visualization_path
    )


    return {

        "mask":
            restored_mask,

        "probability":
            restored_probability,

        "metadata":
            metadata,

        "mask_path":
            mask_path,

        "probability_path":
            probability_path,

        "visualization_path":
            visualization_path,

        "predicted_pixels":
            predicted_pixels,

        "predicted_fraction":
            predicted_fraction,

        "max_probability":
            max_probability,

        "mean_probability":
            mean_probability,
    }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Run the locked final TN3K thyroid "
            "nodule segmentation model on one ultrasound."
        )
    )


    parser.add_argument(

        "image",

        type=str,

        help=(
            "Path to input thyroid ultrasound image."
        ),
    )


    parser.add_argument(

        "--checkpoint",

        type=str,

        default=str(
            DEFAULT_CHECKPOINT
        ),

        help=(
            "Path to tn3k_v1_earlystop_best.pth"
        ),
    )


    parser.add_argument(

        "--output-dir",

        type=str,

        default=str(
            DEFAULT_OUTPUT_DIR
        ),

        help=(
            "Directory where inference outputs "
            "will be saved."
        ),
    )


    args = parser.parse_args()


    infer_single_image(

        image_path=args.image,

        checkpoint_path=args.checkpoint,

        output_dir=args.output_dir,
    )


if __name__ == "__main__":

    main()