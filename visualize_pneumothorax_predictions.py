from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pneumothorax_dataloaders import create_dataloaders
from pneumothorax_model import PneumothoraxResNet34UNet


CHECKPOINT_PATH = (
    Path("checkpoints")
    / "fine_tune_stage_best.pth"
)
OUTPUT_DIRECTORY = Path("prediction_examples")
OUTPUT_PATH = OUTPUT_DIRECTORY / "test_prediction_comparison.png"
PREDICTION_THRESHOLD = 0.35
EXPECTED_TRAINING_STAGE = (
    "balanced_weighted_partial_encoder_finetune"
)
POSITIVE_EXAMPLES = 6
PROGRESS_INTERVAL = 50


def choose_device():
    """Use an NVIDIA GPU when available."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_checkpoint(path, model, device):
    """Load the saved best model without changing it."""

    if not path.is_file():
        raise FileNotFoundError(
            "Best checkpoint was not found at: "
            f"{path.resolve()}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
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


def calculate_dice(prediction, target):
    """Calculate Dice for one binary prediction and target."""

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()
    denominator = prediction.sum() + target.sum()

    if denominator == 0:
        return 1.0

    return float(2.0 * intersection / denominator)


def make_sample(image_id, image, target, prediction):
    """Move one test result to CPU memory for later plotting."""

    image_array = image.squeeze().detach().cpu().numpy()
    target_array = target.squeeze().detach().cpu().numpy()
    prediction_array = (
        prediction.squeeze().detach().cpu().numpy()
    )

    target_array = target_array.astype(bool)
    prediction_array = prediction_array.astype(bool)

    return {
        "image_id": str(image_id),
        "image": image_array,
        "target": target_array,
        "prediction": prediction_array,
        "dice": calculate_dice(
            prediction_array,
            target_array,
        ),
        "predicted_pixels": int(prediction_array.sum()),
    }


@torch.no_grad()
def collect_test_samples(model, data_loader, device):
    """Collect positive cases and two useful negative examples."""

    model.eval()

    positive_samples = []
    correctly_empty_negative = None
    worst_false_positive_negative = None
    total_batches = len(data_loader)

    for batch_number, batch in enumerate(
        data_loader,
        start=1,
    ):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        targets = batch["mask"].to(
            device,
            non_blocking=True,
        ) >= 0.5

        predictions = (
            torch.sigmoid(model(images))
            >= PREDICTION_THRESHOLD
        )

        for index in range(images.shape[0]):
            sample = make_sample(
                batch["image_id"][index],
                images[index],
                targets[index],
                predictions[index],
            )

            if sample["target"].any():
                positive_samples.append(sample)
            elif sample["predicted_pixels"] == 0:
                if correctly_empty_negative is None:
                    correctly_empty_negative = sample
            elif (
                worst_false_positive_negative is None
                or sample["predicted_pixels"]
                > worst_false_positive_negative["predicted_pixels"]
            ):
                worst_false_positive_negative = sample

        if (
            batch_number == 1
            or batch_number % PROGRESS_INTERVAL == 0
            or batch_number == total_batches
        ):
            print(
                f"  Test [{batch_number}/{total_batches}]"
            )

    if not positive_samples:
        raise ValueError(
            "The test set contained no positive images."
        )

    return (
        positive_samples,
        correctly_empty_negative,
        worst_false_positive_negative,
    )


def choose_representative_samples(
    positive_samples,
    correctly_empty_negative,
    worst_false_positive_negative,
):
    """Choose positive cases across the observed Dice range."""

    positive_samples.sort(key=lambda sample: sample["dice"])

    number_to_choose = min(
        POSITIVE_EXAMPLES,
        len(positive_samples),
    )
    selected_indices = np.linspace(
        0,
        len(positive_samples) - 1,
        number_to_choose,
        dtype=int,
    )

    selected = [
        (
            positive_samples[index],
            "Positive test image",
        )
        for index in selected_indices
    ]

    if worst_false_positive_negative is not None:
        selected.append(
            (
                worst_false_positive_negative,
                "Negative: largest false prediction",
            )
        )

    if correctly_empty_negative is not None:
        selected.append(
            (
                correctly_empty_negative,
                "Negative: correctly predicted empty",
            )
        )

    return selected


def show_overlay(axis, image, mask, color, title):
    axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., :3] = color
    overlay[..., 3] = mask.astype(np.float32) * 0.55
    axis.imshow(overlay)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def create_comparison_figure(selected_samples):
    """Create one readable summary figure for all selected cases."""

    rows = len(selected_samples)
    figure, axes = plt.subplots(
        rows,
        4,
        figsize=(14, 3.5 * rows),
        squeeze=False,
    )

    for row, (sample, description) in enumerate(selected_samples):
        image = sample["image"]
        target = sample["target"]
        prediction = sample["prediction"]

        axes[row, 0].imshow(
            image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 0].set_title(
            f"{description}\nX-ray",
            fontsize=9,
        )
        axes[row, 0].axis("off")

        show_overlay(
            axes[row, 1],
            image,
            target,
            (0.0, 1.0, 0.0),
            "Real mask (green)",
        )
        show_overlay(
            axes[row, 2],
            image,
            prediction,
            (1.0, 0.0, 0.0),
            "Prediction (red)",
        )

        overlap = np.logical_and(target, prediction)
        missed = np.logical_and(target, ~prediction)
        false_positive = np.logical_and(~target, prediction)

        comparison = np.zeros(
            (*target.shape, 4),
            dtype=np.float32,
        )
        comparison[missed] = (0.0, 1.0, 0.0, 0.65)
        comparison[false_positive] = (1.0, 0.0, 0.0, 0.65)
        comparison[overlap] = (1.0, 1.0, 0.0, 0.75)

        axes[row, 3].imshow(
            image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 3].imshow(comparison)
        axes[row, 3].set_title(
            "Comparison\n"
            f"Dice = {sample['dice']:.3f}",
            fontsize=9,
        )
        axes[row, 3].axis("off")

    figure.suptitle(
        "Pneumothorax test predictions at threshold "
        f"{PREDICTION_THRESHOLD:.2f}\n"
        "Yellow = correct overlap, green = missed, red = false positive",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    device = choose_device()

    print(f"Device: {device}")
    print(f"Prediction threshold: {PREDICTION_THRESHOLD}")
    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print("Loading test data...")

    test_loader = create_dataloaders()["test"]

    print("Loading model and best checkpoint...")

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=True,
    ).to(device)

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        model,
        device,
    )

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print("Scanning untouched test set...")

    (
        positive_samples,
        correctly_empty_negative,
        worst_false_positive_negative,
    ) = collect_test_samples(
        model,
        test_loader,
        device,
    )

    selected_samples = choose_representative_samples(
        positive_samples,
        correctly_empty_negative,
        worst_false_positive_negative,
    )

    create_comparison_figure(selected_samples)

    print("\nPrediction visualization complete")
    print("---------------------------------")
    print(f"Positive test images scanned: {len(positive_samples)}")
    print(f"Examples shown: {len(selected_samples)}")
    print(f"Saved image: {OUTPUT_PATH.resolve()}")
    print("\nColor guide:")
    print("  Yellow = correctly predicted overlap")
    print("  Green = pneumothorax pixels the model missed")
    print("  Red = pixels predicted incorrectly")


if __name__ == "__main__":
    main()