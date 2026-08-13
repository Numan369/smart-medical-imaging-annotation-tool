import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from check_pneumothorax_augmentation import (
    AugmentedTrainingDataset,
    PairedTrainingAugmentation,
)
from pneumothorax_dataloaders import create_balanced_training_sampler
from pneumothorax_dataset import PneumothoraxDataset
from pneumothorax_model import PneumothoraxResNet34UNet


RANDOM_SEED = 42
IMAGE_SIZE = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
ENCODER_3_WARMUP_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 3
PREDICTION_THRESHOLD = 0.35

DECODER_LEARNING_RATE = 1e-4
ENCODER_4_LEARNING_RATE = 1e-5
ENCODER_3_LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-4
POSITIVE_PIXEL_WEIGHT = 10.0
GRADIENT_CLIP_NORM = 1.0

# Keep this True for the first Colab run. After the smoke-test output passes,
# change it to False to start the complete resumable experiment.
SMOKE_TEST = False
MAX_EPOCHS = 1 if SMOKE_TEST else 10
MAX_TRAINING_BATCHES = 2 if SMOKE_TEST else None
MAX_VALIDATION_BATCHES = 2 if SMOKE_TEST else None
RESUME_IF_AVAILABLE = not SMOKE_TEST

CHECKPOINT_DIRECTORY = Path("checkpoints")
SOURCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "fine_tune_stage_best.pth"
)
LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / (
        "pneumothorax_512_smoke_last.pth"
        if SMOKE_TEST
        else "pneumothorax_512_last.pth"
    )
)
BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / (
        "pneumothorax_512_smoke_best.pth"
        if SMOKE_TEST
        else "pneumothorax_512_best.pth"
    )
)
TRAINING_STAGE = (
    "pneumothorax_512_smoke_test"
    if SMOKE_TEST
    else "pneumothorax_512_augmented_progressive_finetune"
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


class PositiveFocalTverskyLoss(nn.Module):
    """Emphasize false-negative pixels on positive images."""

    def __init__(
        self,
        false_positive_weight=0.30,
        false_negative_weight=0.70,
        gamma=0.75,
        smooth=1.0,
    ):
        super().__init__()
        self.false_positive_weight = false_positive_weight
        self.false_negative_weight = false_negative_weight
        self.gamma = gamma
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
            + self.false_positive_weight * false_positive
            + self.false_negative_weight * false_negative
            + self.smooth
        )

        return ((1.0 - tversky) ** self.gamma).mean()


class UpgradedSegmentationLoss(nn.Module):
    """Combine weighted BCE with positive-only focal Tversky loss."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "positive_pixel_weight",
            torch.tensor([POSITIVE_PIXEL_WEIGHT]),
        )
        self.tversky = PositiveFocalTverskyLoss()

    def components(self, logits, targets):
        targets = targets.to(dtype=logits.dtype)
        bce = functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.positive_pixel_weight.to(
                device=logits.device,
                dtype=logits.dtype,
            ),
        )
        tversky = self.tversky(logits, targets)
        total = 0.40 * bce + 0.60 * tversky
        return total, bce, tversky

    def forward(self, logits, targets):
        return self.components(logits, targets)[0]


def create_data_loaders(device):
    """Create augmented train and untouched validation loaders only."""

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
    training_sampler = create_balanced_training_sampler(
        training_dataset
    )

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


def freeze_all_encoder_parameters(model):
    model.freeze_encoder()


def configure_trainable_parameters(model, epoch):
    """Fine-tune encoder_4 first, then encoder_3 and encoder_4."""

    freeze_all_encoder_parameters(model)

    for parameter in model.encoder_4.parameters():
        parameter.requires_grad = True

    encoder_3_active = epoch > ENCODER_3_WARMUP_EPOCHS

    if encoder_3_active:
        for parameter in model.encoder_3.parameters():
            parameter.requires_grad = True

    return encoder_3_active


def create_optimizer(model):
    """Create stable parameter groups for resuming across both phases."""

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
                "lr": 0.0,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


def activate_epoch_phase(model, optimizer, epoch):
    encoder_3_active = configure_trainable_parameters(model, epoch)

    for group in optimizer.param_groups:
        if group.get("name") == "encoder_3":
            if encoder_3_active and group["lr"] == 0.0:
                group["lr"] = ENCODER_3_LEARNING_RATE

    return encoder_3_active


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
        "tversky_sum": 0.0,
        "dice_sum": 0.0,
        "positive_dice_sum": 0.0,
        "samples": 0,
        "positive_samples": 0,
        "empty_positive_predictions": 0,
        "batches": 0,
    }


def update_metrics(metrics, loss, bce, tversky, logits, targets):
    batch_size = targets.shape[0]
    predictions = torch.sigmoid(logits) >= PREDICTION_THRESHOLD
    binary_targets = targets >= 0.5
    dimensions = tuple(range(1, targets.ndim))

    intersections = (
        predictions & binary_targets
    ).sum(dim=dimensions).float()
    denominators = (
        predictions.sum(dim=dimensions).float()
        + binary_targets.sum(dim=dimensions).float()
    )
    dice = torch.where(
        denominators > 0,
        2.0 * intersections / denominators,
        torch.ones_like(denominators),
    )
    positive_cases = binary_targets.sum(dim=dimensions) > 0
    predicted_areas = predictions.sum(dim=dimensions)

    metrics["loss_sum"] += loss.item() * batch_size
    metrics["bce_sum"] += bce.item() * batch_size
    metrics["tversky_sum"] += tversky.item() * batch_size
    metrics["dice_sum"] += dice.sum().item()
    metrics["samples"] += batch_size
    metrics["batches"] += 1

    if positive_cases.any():
        metrics["positive_dice_sum"] += (
            dice[positive_cases].sum().item()
        )
        metrics["positive_samples"] += int(
            positive_cases.sum().item()
        )
        metrics["empty_positive_predictions"] += int(
            (predicted_areas[positive_cases] == 0).sum().item()
        )


def finalize_metrics(metrics, elapsed_seconds):
    if metrics["samples"] == 0:
        raise ValueError("No images were processed.")

    positive_count = metrics["positive_samples"]
    positive_dice = (
        metrics["positive_dice_sum"] / positive_count
        if positive_count > 0
        else float("nan")
    )
    miss_rate = (
        metrics["empty_positive_predictions"] / positive_count
        if positive_count > 0
        else float("nan")
    )

    return {
        "loss": metrics["loss_sum"] / metrics["samples"],
        "bce": metrics["bce_sum"] / metrics["samples"],
        "tversky": metrics["tversky_sum"] / metrics["samples"],
        "dice": metrics["dice_sum"] / metrics["samples"],
        "positive_dice": positive_dice,
        "empty_positive_predictions": metrics[
            "empty_positive_predictions"
        ],
        "positive_samples": positive_count,
        "positive_miss_rate": miss_rate,
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

    # Preserve the pretrained BatchNorm running statistics. Trainable
    # convolution and affine parameters still receive gradients.
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
                loss, bce, tversky = criterion.components(
                    logits,
                    targets,
                )

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
                    tversky,
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
    print(f"  Validation Dice: {validation['dice']:.6f}")
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
        f"  Training time: {training['seconds'] / 60.0:.1f} min"
    )
    print(
        f"  Validation time: {validation['seconds'] / 60.0:.1f} min"
    )


def validation_improved(validation, best_positive_dice, best_loss):
    tolerance = 1e-8

    if validation["positive_dice"] > best_positive_dice + tolerance:
        return True

    return (
        math.isclose(
            validation["positive_dice"],
            best_positive_dice,
            abs_tol=tolerance,
        )
        and validation["loss"] < best_loss
    )


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_positive_dice,
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
            "best_validation_positive_dice": best_positive_dice,
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
                "encoder_3_warmup_epochs": ENCODER_3_WARMUP_EPOCHS,
                "source_checkpoint": str(SOURCE_CHECKPOINT_PATH),
                "augmentation": "paired_conservative_training_only",
                "loss": "0.40 weighted BCE + 0.60 focal Tversky",
                "tversky_false_positive_weight": 0.30,
                "tversky_false_negative_weight": 0.70,
                "test_split_used": False,
            },
        },
        path,
    )


def load_source_model(model, device):
    if not SOURCE_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "The baseline checkpoint was not found: "
            f"{SOURCE_CHECKPOINT_PATH.resolve()}"
        )

    checkpoint = load_torch_checkpoint(SOURCE_CHECKPOINT_PATH, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def load_resume_state(
    model,
    optimizer,
    scheduler,
    scaler,
    device,
):
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
        "best_positive_dice": checkpoint[
            "best_validation_positive_dice"
        ],
        "best_loss": checkpoint["best_validation_loss"],
        "epochs_without_improvement": checkpoint[
            "epochs_without_improvement"
        ],
    }


def main():
    set_random_seeds(RANDOM_SEED)
    device = choose_device()

    print("Upgraded 512 x 512 pneumothorax fine-tuning")
    print("------------------------------------------")
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
    print("Training augmentation: enabled")
    print("Validation augmentation: disabled")
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
    configure_trainable_parameters(model, epoch=1)

    optimizer = create_optimizer(model)
    criterion = UpgradedSegmentationLoss().to(device)
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
    best_positive_dice = -1.0
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
        best_positive_dice = state["best_positive_dice"]
        best_loss = state["best_loss"]
        epochs_without_improvement = state[
            "epochs_without_improvement"
        ]
        print(f"Resumed from: {LAST_CHECKPOINT_PATH.resolve()}")
        print(f"Next epoch: {start_epoch}")
    else:
        source = load_source_model(model, device)
        print(
            "Loaded baseline checkpoint epoch: "
            f"{source.get('completed_epoch', 'unknown')}"
        )
        print("\nMeasuring the unchanged baseline at 512 x 512...")
        baseline = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            MAX_VALIDATION_BATCHES,
        )
        best_positive_dice = baseline["positive_dice"]
        best_loss = baseline["loss"]
        print(
            "Baseline validation positive-case Dice: "
            f"{best_positive_dice:.6f}"
        )
        print(
            "Baseline empty positive predictions: "
            f"{baseline['empty_positive_predictions']} / "
            f"{baseline['positive_samples']}"
        )
        save_checkpoint(
            BEST_CHECKPOINT_PATH,
            0,
            model,
            optimizer,
            scheduler,
            scaler,
            best_positive_dice,
            best_loss,
            epochs_without_improvement,
            None,
            baseline,
        )

    if start_epoch > MAX_EPOCHS:
        print("\nAll requested epochs are already complete.")
        return

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        encoder_3_active = activate_epoch_phase(
            model,
            optimizer,
            epoch,
        )
        phase = (
            "decoder + encoder_4 + encoder_3"
            if encoder_3_active
            else "decoder + encoder_4 warm-up"
        )
        print(f"\nEpoch {epoch}/{MAX_EPOCHS}")
        print(f"  Fine-tuning phase: {phase}")

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
            best_positive_dice,
            best_loss,
        )

        if improved:
            best_positive_dice = validation["positive_dice"]
            best_loss = validation["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step(validation["positive_dice"])

        save_checkpoint(
            LAST_CHECKPOINT_PATH,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_positive_dice,
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
                best_positive_dice,
                best_loss,
                epochs_without_improvement,
                training,
                validation,
            )
            print(
                "  New best upgraded checkpoint: "
                f"{BEST_CHECKPOINT_PATH.resolve()}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "\nEarly stopping: validation positive-case Dice "
                f"did not improve for {EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    print("\nUpgraded fine-tuning finished.")
    print(
        "Best validation positive-case Dice: "
        f"{best_positive_dice:.6f}"
    )
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH.resolve()}")
    print("The test split was not used.")


if __name__ == "__main__":
    main()
