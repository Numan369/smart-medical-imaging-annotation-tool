import random
import time
from pathlib import Path

import numpy as np
import torch

from pneumothorax_dataloaders import create_dataloaders
from pneumothorax_loss import (
    BCEDiceLoss,
    POSITIVE_PIXEL_WEIGHT,
)
from pneumothorax_model import PneumothoraxResNet34UNet


RANDOM_SEED = 42
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Three total decoder-only epochs are a cautious first balanced run.
# On a CPU, each epoch may take roughly as long as the baseline run.
NUMBER_OF_EPOCHS = 3

# None means process every available batch.
MAX_TRAINING_BATCHES = None
MAX_VALIDATION_BATCHES = None

PROGRESS_INTERVAL = 100
PREDICTION_THRESHOLD = 0.5
RESUME_IF_AVAILABLE = True

CHECKPOINT_DIRECTORY = Path("checkpoints")
LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "balanced_decoder_stage_last.pth"
)
BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "balanced_decoder_stage_best.pth"
)
TRAINING_STAGE = "balanced_weighted_decoder_only"


def set_random_seeds(seed):
    """Make training as reproducible as practical."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    """Use an NVIDIA GPU when available."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def find_validation_loader(data_loaders):
    """Find the validation DataLoader."""

    if "validation" in data_loaders:
        return data_loaders["validation"]

    if "val" in data_loaders:
        return data_loaders["val"]

    raise KeyError(
        "The DataLoader dictionary must contain "
        "'validation' or 'val'."
    )


def calculate_dice_scores(
    logits,
    targets,
    threshold=PREDICTION_THRESHOLD,
):
    """Calculate one thresholded binary Dice score per image."""

    predictions = torch.sigmoid(logits) >= threshold
    targets = targets >= 0.5

    dimensions = tuple(range(1, predictions.ndim))

    intersections = (
        predictions & targets
    ).sum(dim=dimensions).float()

    denominators = (
        predictions.sum(dim=dimensions).float()
        + targets.sum(dim=dimensions).float()
    )

    return torch.where(
        denominators > 0,
        (2.0 * intersections) / denominators,
        torch.ones_like(denominators),
    )


def number_of_batches(data_loader, maximum_batches):
    available_batches = len(data_loader)

    if maximum_batches is None:
        return available_batches

    return min(available_batches, maximum_batches)


def should_print_progress(batch_number, total_batches):
    return (
        batch_number == 1
        or batch_number % PROGRESS_INTERVAL == 0
        or batch_number == total_batches
    )


def batch_positive_dice_text(dice_scores, positive_cases):
    if not positive_cases.any():
        return "n/a"

    positive_dice = dice_scores[positive_cases].mean().item()
    return f"{positive_dice:.6f}"


def create_results(
    total_loss,
    total_dice,
    total_positive_dice,
    total_samples,
    total_positive_samples,
    processed_batches,
    elapsed_seconds,
):
    if total_samples == 0:
        raise ValueError("No samples were processed.")

    if total_positive_samples > 0:
        positive_dice = (
            total_positive_dice
            / total_positive_samples
        )
    else:
        positive_dice = float("nan")

    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
        "positive_dice": positive_dice,
        "batches": processed_batches,
        "samples": total_samples,
        "positive_samples": total_positive_samples,
        "seconds": elapsed_seconds,
    }


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    maximum_batches,
):
    """Train the decoder for one balanced epoch."""

    model.train()

    # Keep the frozen encoder, including its BatchNorm layers,
    # in evaluation mode.
    for encoder_module in model.encoder_modules():
        encoder_module.eval()

    total_batches = number_of_batches(
        data_loader,
        maximum_batches,
    )

    total_loss = 0.0
    total_dice = 0.0
    total_positive_dice = 0.0
    total_samples = 0
    total_positive_samples = 0
    processed_batches = 0

    start_time = time.perf_counter()

    for batch_number, batch in enumerate(
        data_loader,
        start=1,
    ):
        if batch_number > total_batches:
            break

        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        target_masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        output_logits = model(images)
        loss = criterion(output_logits, target_masks)

        if not torch.isfinite(loss):
            raise ValueError(
                "Training produced a non-finite loss."
            )

        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]

        with torch.no_grad():
            dice_scores = calculate_dice_scores(
                output_logits,
                target_masks,
            )

            positive_cases = (
                target_masks.flatten(start_dim=1)
                .any(dim=1)
            )

        total_loss += loss.item() * batch_size
        total_dice += dice_scores.sum().item()
        total_samples += batch_size
        processed_batches += 1

        if positive_cases.any():
            total_positive_dice += (
                dice_scores[positive_cases]
                .sum()
                .item()
            )
            total_positive_samples += int(
                positive_cases.sum().item()
            )

        if should_print_progress(
            batch_number,
            total_batches,
        ):
            elapsed_minutes = (
                time.perf_counter() - start_time
            ) / 60.0

            positive_dice_text = (
                batch_positive_dice_text(
                    dice_scores,
                    positive_cases,
                )
            )

            print(
                f"  Training [{batch_number}/"
                f"{total_batches}] "
                f"loss={loss.item():.6f}, "
                f"dice={dice_scores.mean().item():.6f}, "
                f"positive_dice={positive_dice_text}, "
                f"elapsed={elapsed_minutes:.1f} min"
            )

    elapsed_seconds = time.perf_counter() - start_time

    return create_results(
        total_loss=total_loss,
        total_dice=total_dice,
        total_positive_dice=total_positive_dice,
        total_samples=total_samples,
        total_positive_samples=total_positive_samples,
        processed_batches=processed_batches,
        elapsed_seconds=elapsed_seconds,
    )


@torch.no_grad()
def validate_one_epoch(
    model,
    data_loader,
    criterion,
    device,
    maximum_batches,
):
    """Evaluate without changing model parameters."""

    model.eval()

    total_batches = number_of_batches(
        data_loader,
        maximum_batches,
    )

    total_loss = 0.0
    total_dice = 0.0
    total_positive_dice = 0.0
    total_samples = 0
    total_positive_samples = 0
    processed_batches = 0

    start_time = time.perf_counter()

    for batch_number, batch in enumerate(
        data_loader,
        start=1,
    ):
        if batch_number > total_batches:
            break

        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        target_masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        output_logits = model(images)
        loss = criterion(output_logits, target_masks)

        if not torch.isfinite(loss):
            raise ValueError(
                "Validation produced a non-finite loss."
            )

        dice_scores = calculate_dice_scores(
            output_logits,
            target_masks,
        )

        positive_cases = (
            target_masks.flatten(start_dim=1)
            .any(dim=1)
        )

        batch_size = images.shape[0]

        total_loss += loss.item() * batch_size
        total_dice += dice_scores.sum().item()
        total_samples += batch_size
        processed_batches += 1

        if positive_cases.any():
            total_positive_dice += (
                dice_scores[positive_cases]
                .sum()
                .item()
            )
            total_positive_samples += int(
                positive_cases.sum().item()
            )

        if should_print_progress(
            batch_number,
            total_batches,
        ):
            elapsed_minutes = (
                time.perf_counter() - start_time
            ) / 60.0

            positive_dice_text = (
                batch_positive_dice_text(
                    dice_scores,
                    positive_cases,
                )
            )

            print(
                f"  Validation [{batch_number}/"
                f"{total_batches}] "
                f"loss={loss.item():.6f}, "
                f"dice={dice_scores.mean().item():.6f}, "
                f"positive_dice={positive_dice_text}, "
                f"elapsed={elapsed_minutes:.1f} min"
            )

    elapsed_seconds = time.perf_counter() - start_time

    return create_results(
        total_loss=total_loss,
        total_dice=total_dice,
        total_positive_dice=total_positive_dice,
        total_samples=total_samples,
        total_positive_samples=total_positive_samples,
        processed_batches=processed_batches,
        elapsed_seconds=elapsed_seconds,
    )


def validation_is_better(
    validation_results,
    best_positive_dice,
    best_loss,
):
    """Prefer positive-case Dice, with loss as a tie-breaker."""

    positive_dice = validation_results["positive_dice"]
    validation_loss = validation_results["loss"]

    if positive_dice > best_positive_dice:
        return True

    return (
        positive_dice == best_positive_dice
        and validation_loss < best_loss
    )


def save_checkpoint(
    path,
    completed_epoch,
    model,
    optimizer,
    best_validation_positive_dice,
    best_validation_loss,
    training_results,
    validation_results,
):
    """Save everything needed to continue balanced training."""

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "training_stage": TRAINING_STAGE,
        "completed_epoch": completed_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_positive_dice": (
            best_validation_positive_dice
        ),
        "best_validation_loss": best_validation_loss,
        "training_results": training_results,
        "validation_results": validation_results,
        "configuration": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "positive_pixel_weight": POSITIVE_PIXEL_WEIGHT,
            "prediction_threshold": PREDICTION_THRESHOLD,
        },
    }

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, device):
    """Restore a balanced decoder-training checkpoint."""

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    if checkpoint.get("training_stage") != TRAINING_STAGE:
        raise ValueError(
            "The checkpoint belongs to a different "
            "training stage."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    completed_epoch = checkpoint["completed_epoch"]

    best_positive_dice = checkpoint.get(
        "best_validation_positive_dice",
        -1.0,
    )
    best_loss = checkpoint.get(
        "best_validation_loss",
        float("inf"),
    )

    return completed_epoch + 1, best_positive_dice, best_loss


def print_epoch_summary(training_results, validation_results):
    print("\nEpoch summary")
    print(
        "  Training loss: "
        f"{training_results['loss']:.6f}"
    )
    print(
        "  Training Dice: "
        f"{training_results['dice']:.6f}"
    )
    print(
        "  Training positive-case Dice: "
        f"{training_results['positive_dice']:.6f}"
    )
    print(
        "  Training positive images: "
        f"{training_results['positive_samples']}"
    )
    print(
        "  Validation loss: "
        f"{validation_results['loss']:.6f}"
    )
    print(
        "  Validation Dice: "
        f"{validation_results['dice']:.6f}"
    )
    print(
        "  Validation positive-case Dice: "
        f"{validation_results['positive_dice']:.6f}"
    )
    print(
        "  Validation positive images: "
        f"{validation_results['positive_samples']}"
    )
    print(
        "  Training time: "
        f"{training_results['seconds'] / 60.0:.1f} min"
    )
    print(
        "  Validation time: "
        f"{validation_results['seconds'] / 60.0:.1f} min"
    )


def main():
    set_random_seeds(RANDOM_SEED)

    device = choose_device()

    print("Balanced decoder training stage")
    print("-------------------------------")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_loaders = create_dataloaders()
    training_loader = data_loaders["train"]
    validation_loader = find_validation_loader(data_loaders)

    print(
        "Training batches available: "
        f"{len(training_loader)}"
    )
    print(
        "Validation batches available: "
        f"{len(validation_loader)}"
    )
    print(
        "Positive-pixel BCE weight: "
        f"{POSITIVE_PIXEL_WEIGHT:.1f}"
    )
    print("Loading pretrained segmentation model...")

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=True,
        freeze_encoder=True,
    ).to(device)

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        positive_pixel_weight=POSITIVE_PIXEL_WEIGHT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    start_epoch = 1
    best_validation_positive_dice = -1.0
    best_validation_loss = float("inf")

    if (
        RESUME_IF_AVAILABLE
        and LAST_CHECKPOINT_PATH.exists()
    ):
        (
            start_epoch,
            best_validation_positive_dice,
            best_validation_loss,
        ) = load_checkpoint(
            LAST_CHECKPOINT_PATH,
            model,
            optimizer,
            device,
        )

        print(
            "Resumed from: "
            f"{LAST_CHECKPOINT_PATH.resolve()}"
        )
        print(f"Next epoch: {start_epoch}")

    if start_epoch > NUMBER_OF_EPOCHS:
        print(
            "\nThe requested balanced decoder-training "
            "epochs are already complete."
        )
        return

    active_epoch = start_epoch

    try:
        for epoch_number in range(
            start_epoch,
            NUMBER_OF_EPOCHS + 1,
        ):
            active_epoch = epoch_number

            print(
                f"\nEpoch {epoch_number}/"
                f"{NUMBER_OF_EPOCHS}"
            )

            training_results = train_one_epoch(
                model=model,
                data_loader=training_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                maximum_batches=MAX_TRAINING_BATCHES,
            )

            validation_results = validate_one_epoch(
                model=model,
                data_loader=validation_loader,
                criterion=criterion,
                device=device,
                maximum_batches=MAX_VALIDATION_BATCHES,
            )

            print_epoch_summary(
                training_results,
                validation_results,
            )

            validation_improved = validation_is_better(
                validation_results,
                best_validation_positive_dice,
                best_validation_loss,
            )

            if validation_improved:
                best_validation_positive_dice = (
                    validation_results["positive_dice"]
                )
                best_validation_loss = (
                    validation_results["loss"]
                )

            save_checkpoint(
                path=LAST_CHECKPOINT_PATH,
                completed_epoch=epoch_number,
                model=model,
                optimizer=optimizer,
                best_validation_positive_dice=(
                    best_validation_positive_dice
                ),
                best_validation_loss=best_validation_loss,
                training_results=training_results,
                validation_results=validation_results,
            )

            print(
                "  Latest checkpoint: "
                f"{LAST_CHECKPOINT_PATH.resolve()}"
            )

            if validation_improved:
                save_checkpoint(
                    path=BEST_CHECKPOINT_PATH,
                    completed_epoch=epoch_number,
                    model=model,
                    optimizer=optimizer,
                    best_validation_positive_dice=(
                        best_validation_positive_dice
                    ),
                    best_validation_loss=best_validation_loss,
                    training_results=training_results,
                    validation_results=validation_results,
                )

                print(
                    "  New best positive-case Dice checkpoint: "
                    f"{BEST_CHECKPOINT_PATH.resolve()}"
                )

    except KeyboardInterrupt:
        # Preserve partial parameter updates. The interrupted epoch
        # will start again when the script is rerun.
        save_checkpoint(
            path=LAST_CHECKPOINT_PATH,
            completed_epoch=active_epoch - 1,
            model=model,
            optimizer=optimizer,
            best_validation_positive_dice=(
                best_validation_positive_dice
            ),
            best_validation_loss=best_validation_loss,
            training_results=None,
            validation_results=None,
        )

        print("\nTraining interrupted safely.")
        print(
            "Partial progress saved to: "
            f"{LAST_CHECKPOINT_PATH.resolve()}"
        )
        return

    print("\nBalanced decoder-training stage finished.")
    print(
        "Best validation positive-case Dice: "
        f"{best_validation_positive_dice:.6f}"
    )
    print(
        "Validation loss at that checkpoint: "
        f"{best_validation_loss:.6f}"
    )
    print(
        "Best checkpoint: "
        f"{BEST_CHECKPOINT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()
