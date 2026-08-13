import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, WeightedRandomSampler

from check_pneumothorax_augmentation import (
    AugmentedTrainingDataset,
    PairedTrainingAugmentation,
)
from pneumothorax_dataset import PneumothoraxDataset
from pneumothorax_model import PneumothoraxResNet34UNet


RANDOM_SEED = 42
IMAGE_SIZE = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
EARLY_STOPPING_PATIENCE = 3
PREDICTION_THRESHOLD = 0.35

# The earlier run sampled approximately 50% positive images. This controlled
# experiment lowers that to 35%, giving the model more negative examples while
# retaining more positives than their natural training prevalence (~22%).
TRAINING_POSITIVE_FRACTION = 0.35

DECODER_LEARNING_RATE = 1e-4
ENCODER_4_LEARNING_RATE = 1e-5
ENCODER_3_LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-4
POSITIVE_PIXEL_WEIGHT = 4.0
GRADIENT_CLIP_NORM = 1.0

BCE_WEIGHT = 0.45
POSITIVE_TVERSKY_WEIGHT = 0.35
NEGATIVE_BCE_WEIGHT = 0.20
TVERSKY_FALSE_POSITIVE_WEIGHT = 0.50
TVERSKY_FALSE_NEGATIVE_WEIGHT = 0.50
TVERSKY_GAMMA = 0.75

# Keep this True for the first Colab run. After the smoke test succeeds,
# change it to False in the temporary Colab copy for the full experiment.
SMOKE_TEST = True
MAX_EPOCHS = 1 if SMOKE_TEST else 8
MAX_TRAINING_BATCHES = 2 if SMOKE_TEST else None
MAX_VALIDATION_BATCHES = 2 if SMOKE_TEST else None
RESUME_IF_AVAILABLE = not SMOKE_TEST

CHECKPOINT_DIRECTORY = Path("checkpoints")
SOURCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "pneumothorax_512_best.pth"
)
LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / (
        "pneumothorax_512_negative_aware_smoke_last.pth"
        if SMOKE_TEST
        else "pneumothorax_512_negative_aware_last.pth"
    )
)
BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / (
        "pneumothorax_512_negative_aware_smoke_best.pth"
        if SMOKE_TEST
        else "pneumothorax_512_negative_aware_best.pth"
    )
)
TRAINING_STAGE = (
    "pneumothorax_512_negative_aware_smoke_test"
    if SMOKE_TEST
    else "pneumothorax_512_negative_aware_finetune"
)


def set_random_seeds(seed):
    """Make the run reproducible as far as the hardware permits."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_torch_checkpoint(path, device):
    """Load checkpoints on both older and newer PyTorch versions."""

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def dataset_labels(dataset):
    """Read image-level labels without loading DICOM pixel data."""

    labels = torch.tensor(
        [int(row["HasPneumothorax"]) for row in dataset.rows],
        dtype=torch.long,
    )

    unexpected = set(labels.tolist()) - {0, 1}
    if unexpected:
        raise ValueError(
            f"Training labels must be 0 or 1. Found: {sorted(unexpected)}"
        )

    return labels


def create_training_sampler(dataset):
    """Sample a reproducible 35% positive and 65% negative mixture."""

    labels = dataset_labels(dataset)
    positive_count = int(labels.sum().item())
    negative_count = len(labels) - positive_count

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Training requires both positive and negative images."
        )

    negative_fraction = 1.0 - TRAINING_POSITIVE_FRACTION
    class_weights = torch.tensor(
        [
            negative_fraction / negative_count,
            TRAINING_POSITIVE_FRACTION / positive_count,
        ],
        dtype=torch.double,
    )
    sample_weights = class_weights[labels]

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


class PositiveFocalTverskyLoss(nn.Module):
    """Measure overlap on positive images with equal FP and FN pressure."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        targets = targets.to(dtype=probabilities.dtype)
        dimensions = tuple(range(1, targets.ndim))
        positive_images = targets.sum(dim=dimensions) > 0

        if not positive_images.any():
            return logits.sum() * 0.0

        probabilities = probabilities[positive_images]
        targets = targets[positive_images]
        dimensions = tuple(range(1, targets.ndim))

        true_positive = (probabilities * targets).sum(dim=dimensions)
        false_positive = (
            probabilities * (1.0 - targets)
        ).sum(dim=dimensions)
        false_negative = (
            (1.0 - probabilities) * targets
        ).sum(dim=dimensions)

        tversky = (
            true_positive + self.smooth
        ) / (
            true_positive
            + TVERSKY_FALSE_POSITIVE_WEIGHT * false_positive
            + TVERSKY_FALSE_NEGATIVE_WEIGHT * false_negative
            + self.smooth
        )

        return ((1.0 - tversky) ** TVERSKY_GAMMA).mean()


class NegativeAwareSegmentationLoss(nn.Module):
    """Combine localization learning with explicit healthy-image pressure."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "positive_pixel_weight",
            torch.tensor([POSITIVE_PIXEL_WEIGHT]),
        )
        self.positive_tversky = PositiveFocalTverskyLoss()

    def components(self, logits, targets):
        targets = targets.to(dtype=logits.dtype)
        positive_pixel_weight = self.positive_pixel_weight.to(
            device=logits.device,
            dtype=logits.dtype,
        )

        bce = functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=positive_pixel_weight,
        )
        positive_tversky = self.positive_tversky(logits, targets)

        dimensions = tuple(range(1, targets.ndim))
        negative_images = targets.sum(dim=dimensions) == 0
        if negative_images.any():
            negative_bce = functional.binary_cross_entropy_with_logits(
                logits[negative_images],
                targets[negative_images],
            )
        else:
            negative_bce = logits.sum() * 0.0

        total = (
            BCE_WEIGHT * bce
            + POSITIVE_TVERSKY_WEIGHT * positive_tversky
            + NEGATIVE_BCE_WEIGHT * negative_bce
        )
        return total, bce, positive_tversky, negative_bce

    def forward(self, logits, targets):
        return self.components(logits, targets)[0]


def create_data_loaders(device):
    """Create augmented training and untouched validation loaders only."""

    training_base = PneumothoraxDataset(
        split="train",
        image_size=IMAGE_SIZE,
    )
    validation_dataset = PneumothoraxDataset(
        split="validation",
        image_size=IMAGE_SIZE,
    )
    training_dataset = AugmentedTrainingDataset(
        training_base,
        PairedTrainingAugmentation(),
    )
    training_sampler = create_training_sampler(training_dataset)

    number_of_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        sampler=training_sampler,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return training_loader, validation_loader


def configure_trainable_parameters(model):
    """Continue from the source checkpoint's decoder + blocks 3 and 4 phase."""

    model.freeze_encoder()

    for parameter in model.encoder_3.parameters():
        parameter.requires_grad = True
    for parameter in model.encoder_4.parameters():
        parameter.requires_grad = True


def create_optimizer(model):
    """Use the same parameter groups and learning rates as the earlier run."""

    encoder_3_parameters = list(model.encoder_3.parameters())
    encoder_4_parameters = list(model.encoder_4.parameters())
    all_encoder_ids = {
        id(parameter) for parameter in model.encoder_parameters()
    }
    decoder_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in all_encoder_ids
    ]

    return torch.optim.AdamW(
        [
            {
                "name": "decoder",
                "params": decoder_parameters,
                "lr": DECODER_LEARNING_RATE,
            },
            {
                "name": "encoder_4",
                "params": encoder_4_parameters,
                "lr": ENCODER_4_LEARNING_RATE,
            },
            {
                "name": "encoder_3",
                "params": encoder_3_parameters,
                "lr": ENCODER_3_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


def available_batches(data_loader, maximum_batches):
    if maximum_batches is None:
        return len(data_loader)
    return min(len(data_loader), maximum_batches)


def should_print_progress(batch_number, total_batches):
    return (
        batch_number == 1
        or batch_number % 100 == 0
        or batch_number == total_batches
    )


def empty_metrics():
    return {
        "loss_sum": 0.0,
        "bce_sum": 0.0,
        "positive_tversky_sum": 0.0,
        "negative_bce_sum": 0.0,
        "dice_sum": 0.0,
        "positive_dice_sum": 0.0,
        "negative_predicted_fraction_sum": 0.0,
        "samples": 0,
        "positive_samples": 0,
        "negative_samples": 0,
        "empty_positive_predictions": 0,
        "empty_negative_predictions": 0,
        "batches": 0,
    }


def update_metrics(
    metrics,
    loss,
    bce,
    positive_tversky,
    negative_bce,
    logits,
    targets,
):
    batch_size = targets.shape[0]
    predictions = torch.sigmoid(logits) >= PREDICTION_THRESHOLD
    binary_targets = targets >= 0.5
    dimensions = tuple(range(1, targets.ndim))

    intersections = (
        predictions & binary_targets
    ).sum(dim=dimensions).float()
    predicted_areas = predictions.sum(dim=dimensions).float()
    target_areas = binary_targets.sum(dim=dimensions).float()
    denominators = predicted_areas + target_areas
    dice = torch.where(
        denominators > 0,
        2.0 * intersections / denominators,
        torch.ones_like(denominators),
    )
    positive_cases = target_areas > 0
    negative_cases = ~positive_cases

    metrics["loss_sum"] += loss.item() * batch_size
    metrics["bce_sum"] += bce.item() * batch_size
    metrics["positive_tversky_sum"] += (
        positive_tversky.item() * batch_size
    )
    metrics["negative_bce_sum"] += negative_bce.item() * batch_size
    metrics["dice_sum"] += dice.sum().item()
    metrics["samples"] += batch_size
    metrics["batches"] += 1

    if positive_cases.any():
        positive_count = int(positive_cases.sum().item())
        metrics["positive_dice_sum"] += (
            dice[positive_cases].sum().item()
        )
        metrics["positive_samples"] += positive_count
        metrics["empty_positive_predictions"] += int(
            (predicted_areas[positive_cases] == 0).sum().item()
        )

    if negative_cases.any():
        negative_count = int(negative_cases.sum().item())
        metrics["negative_samples"] += negative_count
        metrics["empty_negative_predictions"] += int(
            (predicted_areas[negative_cases] == 0).sum().item()
        )
        pixels_per_image = float(IMAGE_SIZE * IMAGE_SIZE)
        metrics["negative_predicted_fraction_sum"] += (
            predicted_areas[negative_cases].sum().item()
            / pixels_per_image
        )


def harmonic_mean(first, second):
    if first <= 0.0 or second <= 0.0:
        return 0.0
    return 2.0 * first * second / (first + second)


def finalize_metrics(metrics, elapsed_seconds):
    if metrics["samples"] == 0:
        raise ValueError("No images were processed.")

    positive_count = metrics["positive_samples"]
    negative_count = metrics["negative_samples"]
    positive_dice = (
        metrics["positive_dice_sum"] / positive_count
        if positive_count > 0
        else 0.0
    )
    negative_empty_accuracy = (
        metrics["empty_negative_predictions"] / negative_count
        if negative_count > 0
        else 0.0
    )
    selection_score = harmonic_mean(
        positive_dice,
        negative_empty_accuracy,
    )

    return {
        "loss": metrics["loss_sum"] / metrics["samples"],
        "bce": metrics["bce_sum"] / metrics["samples"],
        "positive_tversky": (
            metrics["positive_tversky_sum"] / metrics["samples"]
        ),
        "negative_bce": (
            metrics["negative_bce_sum"] / metrics["samples"]
        ),
        "dice": metrics["dice_sum"] / metrics["samples"],
        "positive_dice": positive_dice,
        "negative_empty_accuracy": negative_empty_accuracy,
        "selection_score": selection_score,
        "empty_positive_predictions": metrics[
            "empty_positive_predictions"
        ],
        "positive_samples": positive_count,
        "positive_miss_rate": (
            metrics["empty_positive_predictions"] / positive_count
            if positive_count > 0
            else 0.0
        ),
        "false_positive_negative_images": (
            negative_count - metrics["empty_negative_predictions"]
        ),
        "negative_samples": negative_count,
        "negative_mean_predicted_fraction": (
            metrics["negative_predicted_fraction_sum"] / negative_count
            if negative_count > 0
            else 0.0
        ),
        "samples": metrics["samples"],
        "batches": metrics["batches"],
        "seconds": elapsed_seconds,
    }


def run_epoch(
    model,
    data_loader,
    criterion,
    device,
    maximum_batches,
    optimizer=None,
    scaler=None,
):
    is_training = optimizer is not None
    model.train(is_training)

    # Preserve pretrained encoder BatchNorm running statistics.
    for encoder_module in model.encoder_modules():
        encoder_module.eval()

    total_batches = available_batches(data_loader, maximum_batches)
    metrics = empty_metrics()
    start_time = time.perf_counter()
    use_amp = device.type == "cuda"

    if is_training:
        optimizer.zero_grad(set_to_none=True)

    gradient_context = (
        torch.enable_grad() if is_training else torch.no_grad()
    )

    with gradient_context:
        for batch_number, batch in enumerate(data_loader, start=1):
            if batch_number > total_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                (
                    loss,
                    bce,
                    positive_tversky,
                    negative_bce,
                ) = criterion.components(logits, targets)

            if not torch.isfinite(loss):
                raise ValueError("A non-finite loss was produced.")

            if is_training:
                scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
                scaler.scale(scaled_loss).backward()
                accumulation_complete = (
                    batch_number % GRADIENT_ACCUMULATION_STEPS == 0
                    or batch_number == total_batches
                )

                if accumulation_complete:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ),
                        GRADIENT_CLIP_NORM,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                update_metrics(
                    metrics,
                    loss,
                    bce,
                    positive_tversky,
                    negative_bce,
                    logits,
                    targets,
                )

            if should_print_progress(batch_number, total_batches):
                elapsed_minutes = (
                    time.perf_counter() - start_time
                ) / 60.0
                phase = "Training" if is_training else "Validation"
                print(
                    f"  {phase} [{batch_number}/{total_batches}] "
                    f"loss={loss.item():.6f}, "
                    f"elapsed={elapsed_minutes:.1f} min"
                )

    return finalize_metrics(
        metrics,
        time.perf_counter() - start_time,
    )


def print_epoch_results(training, validation):
    print("\nEpoch summary")
    print(f"  Training loss: {training['loss']:.6f}")
    print(
        "  Training positive-case Dice: "
        f"{training['positive_dice']:.6f}"
    )
    print(f"  Validation loss: {validation['loss']:.6f}")
    print(
        "  Validation positive-case Dice: "
        f"{validation['positive_dice']:.6f}"
    )
    print(
        "  Validation empty positive predictions: "
        f"{validation['empty_positive_predictions']} / "
        f"{validation['positive_samples']} "
        f"({100.0 * validation['positive_miss_rate']:.2f}%)"
    )
    print(
        "  Validation false-positive negative images: "
        f"{validation['false_positive_negative_images']} / "
        f"{validation['negative_samples']} "
        f"({100.0 * (1.0 - validation['negative_empty_accuracy']):.2f}%)"
    )
    print(
        "  Validation negative empty-mask accuracy: "
        f"{validation['negative_empty_accuracy']:.6f}"
    )
    print(
        "  Validation mean predicted area on negatives: "
        f"{100.0 * validation['negative_mean_predicted_fraction']:.4f}%"
    )
    print(
        "  Validation joint selection score: "
        f"{validation['selection_score']:.6f}"
    )
    print(
        f"  Training time: {training['seconds'] / 60.0:.1f} min"
    )
    print(
        f"  Validation time: {validation['seconds'] / 60.0:.1f} min"
    )


def validation_improved(validation, best_score, best_loss):
    tolerance = 1e-8
    score = validation["selection_score"]

    if score > best_score + tolerance:
        return True

    return (
        math.isclose(score, best_score, abs_tol=tolerance)
        and validation["loss"] < best_loss
    )


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_score,
    best_loss,
    epochs_without_improvement,
    training_results,
    validation_results,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "training_stage": TRAINING_STAGE,
            "completed_epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_selection_score": best_score,
            "best_validation_loss": best_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "training_results": training_results,
            "validation_results": validation_results,
            "configuration": {
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": (
                    GRADIENT_ACCUMULATION_STEPS
                ),
                "prediction_threshold": PREDICTION_THRESHOLD,
                "maximum_epochs": MAX_EPOCHS,
                "training_positive_fraction": (
                    TRAINING_POSITIVE_FRACTION
                ),
                "source_checkpoint": str(SOURCE_CHECKPOINT_PATH),
                "augmentation": "paired_conservative_training_only",
                "loss": (
                    "0.45 weighted BCE + 0.35 symmetric positive "
                    "focal Tversky + 0.20 negative-only BCE"
                ),
                "positive_pixel_weight": POSITIVE_PIXEL_WEIGHT,
                "tversky_false_positive_weight": (
                    TVERSKY_FALSE_POSITIVE_WEIGHT
                ),
                "tversky_false_negative_weight": (
                    TVERSKY_FALSE_NEGATIVE_WEIGHT
                ),
                "checkpoint_selection": (
                    "harmonic mean of validation positive Dice and "
                    "negative empty-mask accuracy"
                ),
                "test_split_used": False,
            },
        },
        path,
    )


def load_source_model(model, device):
    if not SOURCE_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "The source checkpoint was not found: "
            f"{SOURCE_CHECKPOINT_PATH.resolve()}"
        )

    checkpoint = load_torch_checkpoint(SOURCE_CHECKPOINT_PATH, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def load_resume_state(model, optimizer, scheduler, scaler, device):
    checkpoint = load_torch_checkpoint(LAST_CHECKPOINT_PATH, device)

    if checkpoint.get("training_stage") != TRAINING_STAGE:
        raise ValueError(
            "The resume checkpoint belongs to a different stage."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))

    return {
        "start_epoch": checkpoint["completed_epoch"] + 1,
        "best_score": checkpoint["best_validation_selection_score"],
        "best_loss": checkpoint["best_validation_loss"],
        "epochs_without_improvement": checkpoint[
            "epochs_without_improvement"
        ],
    }


def main():
    set_random_seeds(RANDOM_SEED)
    device = choose_device()

    print("Negative-aware 512 x 512 pneumothorax fine-tuning")
    print("------------------------------------------------")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(
        "Run mode: "
        + ("SMOKE TEST" if SMOKE_TEST else "FULL TRAINING")
    )
    print(f"Batch size: {BATCH_SIZE}")
    print(
        "Effective batch size: "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )
    print(
        "Target training mixture: "
        f"{100.0 * TRAINING_POSITIVE_FRACTION:.0f}% positive / "
        f"{100.0 * (1.0 - TRAINING_POSITIVE_FRACTION):.0f}% negative"
    )
    print("Training augmentation: enabled")
    print("Validation augmentation: disabled")
    print("Fine-tuning phase: decoder + encoder_4 + encoder_3")
    print("Dataset splits: train and validation only")
    print("Test split: not created or accessed")
    print(f"Source checkpoint: {SOURCE_CHECKPOINT_PATH.resolve()}")

    training_loader, validation_loader = create_data_loaders(device)
    print(f"Training batches: {len(training_loader):,}")
    print(f"Validation batches: {len(validation_loader):,}")

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=True,
    ).to(device)
    configure_trainable_parameters(model)
    optimizer = create_optimizer(model)
    criterion = NegativeAwareSegmentationLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda"
    )

    start_epoch = 1
    best_score = -1.0
    best_loss = float("inf")
    epochs_without_improvement = 0

    if RESUME_IF_AVAILABLE and LAST_CHECKPOINT_PATH.is_file():
        state = load_resume_state(
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        start_epoch = state["start_epoch"]
        best_score = state["best_score"]
        best_loss = state["best_loss"]
        epochs_without_improvement = state[
            "epochs_without_improvement"
        ]
        print(f"Resumed from: {LAST_CHECKPOINT_PATH.resolve()}")
        print(f"Next epoch: {start_epoch}")
    else:
        source = load_source_model(model, device)
        print(
            "Loaded source checkpoint epoch: "
            f"{source.get('completed_epoch', 'unknown')}"
        )
        print("\nMeasuring the unchanged source model...")
        baseline = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            MAX_VALIDATION_BATCHES,
        )
        best_score = baseline["selection_score"]
        best_loss = baseline["loss"]
        print(
            "Source validation positive-case Dice: "
            f"{baseline['positive_dice']:.6f}"
        )
        print(
            "Source negative empty-mask accuracy: "
            f"{baseline['negative_empty_accuracy']:.6f}"
        )
        print(
            "Source joint selection score: "
            f"{baseline['selection_score']:.6f}"
        )
        save_checkpoint(
            BEST_CHECKPOINT_PATH,
            0,
            model,
            optimizer,
            scheduler,
            scaler,
            best_score,
            best_loss,
            epochs_without_improvement,
            None,
            baseline,
        )

    if start_epoch > MAX_EPOCHS:
        print("\nAll requested epochs are already complete.")
        return

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{MAX_EPOCHS}")
        print("  Fine-tuning phase: decoder + encoder_4 + encoder_3")

        training = run_epoch(
            model,
            training_loader,
            criterion,
            device,
            MAX_TRAINING_BATCHES,
            optimizer=optimizer,
            scaler=scaler,
        )
        validation = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            MAX_VALIDATION_BATCHES,
        )
        print_epoch_results(training, validation)

        improved = validation_improved(
            validation,
            best_score,
            best_loss,
        )
        if improved:
            best_score = validation["selection_score"]
            best_loss = validation["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step(validation["selection_score"])

        save_checkpoint(
            LAST_CHECKPOINT_PATH,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_score,
            best_loss,
            epochs_without_improvement,
            training,
            validation,
        )
        print(f"  Latest checkpoint: {LAST_CHECKPOINT_PATH.resolve()}")

        if improved:
            save_checkpoint(
                BEST_CHECKPOINT_PATH,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_score,
                best_loss,
                epochs_without_improvement,
                training,
                validation,
            )
            print(
                "  New best negative-aware checkpoint: "
                f"{BEST_CHECKPOINT_PATH.resolve()}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "\nEarly stopping: validation joint selection score "
                f"did not improve for {EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    print("\nNegative-aware fine-tuning finished.")
    print(f"Best validation joint score: {best_score:.6f}")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH.resolve()}")
    print("The test split was not used.")


if __name__ == "__main__":
    main()
