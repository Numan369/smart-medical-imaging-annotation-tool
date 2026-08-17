"""Compare locked SIIM V1 predictions with PTX-498 PNG ground truths."""

from argparse import ArgumentParser
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from pneumothorax_dataset import resize_image_and_mask
from pneumothorax_model import PneumothoraxResNet34UNet


DEFAULT_CHECKPOINT = Path("checkpoints/pneumothorax_512_negative_aware_best.pth")
EXPECTED_STAGE = "pneumothorax_512_negative_aware_finetune"
EXPECTED_EPOCH = 4
IMAGE_SIZE = 512
THRESHOLD = 0.35


def arguments():
    parser = ArgumentParser(
        description="Compare V1 with one independent PTX-498 image/mask pair."
    )
    parser.add_argument(
        "dataset_directory",
        type=Path,
        help="Folder containing files such as 1.1.img.png and 1.2.mask.png.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional exact *.img.png file; otherwise a random pair is used.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--save", type=Path, help="Optional output PNG path.")
    return parser.parse_args()


def mask_path_for(image_path):
    name = image_path.name
    if not name.endswith(".img.png"):
        raise ValueError(f"Expected a filename ending in .img.png: {image_path}")

    prefix = name[: -len(".img.png")]
    parts = prefix.rsplit(".", 1)
    if len(parts) == 2 and parts[1].isdigit():
        mask_name = f"{parts[0]}.{int(parts[1]) + 1}.mask.png"
    else:
        mask_name = f"{prefix}.mask.png"
    return image_path.with_name(mask_name)


def discover_pairs(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"PTX directory was not found: {directory.resolve()}")

    pairs = []
    for image_path in sorted(directory.rglob("*.img.png")):
        mask_path = mask_path_for(image_path)
        if mask_path.is_file():
            pairs.append((image_path, mask_path))

    if not pairs:
        raise FileNotFoundError(
            "No matching *.img.png and *.mask.png pairs were found."
        )
    return pairs


def load_checkpoint(path, device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {path.resolve()}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if checkpoint.get("training_stage") != EXPECTED_STAGE:
        raise ValueError("The checkpoint is not the locked V1 negative-aware model.")
    if checkpoint.get("completed_epoch") != EXPECTED_EPOCH:
        raise ValueError("The checkpoint is not the selected V1 epoch-4 model.")

    configuration = checkpoint.get("configuration", {})
    if int(configuration.get("image_size", -1)) != IMAGE_SIZE:
        raise ValueError("The checkpoint does not use the expected 512 image size.")
    if not np.isclose(
        float(configuration.get("prediction_threshold", -1)), THRESHOLD
    ):
        raise ValueError("The checkpoint does not use the locked 0.35 threshold.")
    return checkpoint


def load_pair(image_path, mask_path):
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0

    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shapes differ: {image.shape} versus {mask.shape}."
        )
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum <= minimum:
        raise ValueError("The X-ray has no usable intensity range.")

    # Match the min-max normalization used for the SIIM DICOM model input.
    image = ((image - minimum) / (maximum - minimum)).astype(np.float32)
    image_tensor, mask_tensor = resize_image_and_mask(
        image, mask.astype(np.uint8), IMAGE_SIZE
    )
    return image_tensor, mask_tensor


def metrics(prediction, actual):
    prediction = prediction.astype(bool)
    actual = actual.astype(bool)
    intersection = int(np.logical_and(prediction, actual).sum())
    predicted_area = int(prediction.sum())
    actual_area = int(actual.sum())
    union = int(np.logical_or(prediction, actual).sum())
    dice_denominator = predicted_area + actual_area
    return {
        "dice": 2 * intersection / dice_denominator if dice_denominator else 1.0,
        "precision": intersection / predicted_area if predicted_area else 0.0,
        "recall": intersection / actual_area if actual_area else 0.0,
        "iou": intersection / union if union else 1.0,
        "actual_pixels": actual_area,
        "predicted_pixels": predicted_area,
    }


def overlay(image, mask, colour):
    rgb = np.repeat(image[..., None], 3, axis=2)
    rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.asarray(colour)
    return np.clip(rgb, 0, 1)


def comparison(image, actual, predicted):
    rgb = np.repeat(image[..., None], 3, axis=2)
    overlap = actual & predicted
    missed = actual & ~predicted
    extra = ~actual & predicted
    rgb[overlap] = 0.45 * rgb[overlap] + 0.55 * np.array([0, 1, 0])
    rgb[missed] = 0.45 * rgb[missed] + 0.55 * np.array([1, 0, 0])
    rgb[extra] = 0.45 * rgb[extra] + 0.55 * np.array([0, 0.4, 1])
    return np.clip(rgb, 0, 1)


def main():
    args = arguments()
    pairs = discover_pairs(args.dataset_directory)

    if args.image:
        image_path = args.image.resolve()
        mask_path = mask_path_for(image_path)
        if not mask_path.is_file():
            raise FileNotFoundError(f"Matching mask was not found: {mask_path}")
    else:
        image_path, mask_path = random.choice(pairs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    image_tensor, mask_tensor = load_pair(image_path, mask_path)

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False, freeze_encoder=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        probabilities = torch.sigmoid(model(image_tensor.unsqueeze(0).to(device)))
        predicted_tensor = probabilities >= THRESHOLD

    image = image_tensor.squeeze(0).numpy()
    actual = mask_tensor.squeeze(0).numpy() >= 0.5
    predicted = predicted_tensor.squeeze().cpu().numpy()
    result = metrics(predicted, actual)

    print("PTX-498 external comparison")
    print("Discovered image/mask pairs:", len(pairs))
    print("Selected X-ray:", image_path)
    print("Ground-truth mask:", mask_path)
    print("Device:", device)
    print("Checkpoint: locked V1 epoch 4")
    print("Fixed threshold:", THRESHOLD)
    print(f"Dice: {result['dice']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"IoU: {result['iou']:.4f}")

    figure, axes = plt.subplots(1, 4, figsize=(20, 6))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("External PTX X-ray")
    axes[1].imshow(overlay(image, actual, [1, 0.85, 0]))
    axes[1].set_title("Actual PTX mask")
    axes[2].imshow(overlay(image, predicted, [0, 0.8, 1]))
    axes[2].set_title(f"V1 prediction (threshold {THRESHOLD:.2f})")
    axes[3].imshow(comparison(image, actual, predicted))
    axes[3].set_title("Green=overlap, Red=missed, Blue=extra")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"{image_path.name} | Dice {result['dice']:.4f} | "
        f"Precision {result['precision']:.4f} | Recall {result['recall']:.4f} | "
        f"IoU {result['iou']:.4f}"
    )
    figure.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=160, bbox_inches="tight")
        print("Saved:", args.save.resolve())
    plt.show()


if __name__ == "__main__":
    main()
