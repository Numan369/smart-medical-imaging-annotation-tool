import time
from pathlib import Path

import torch

from pneumothorax_dataloaders import create_dataloaders
from pneumothorax_loss import BCEDiceLoss, POSITIVE_PIXEL_WEIGHT
from pneumothorax_model import PneumothoraxResNet34UNet
from train_pneumothorax_balanced import (
    batch_positive_dice_text,
    choose_device,
    create_results,
    find_validation_loader,
    number_of_batches,
    print_epoch_summary,
    set_random_seeds,
    should_print_progress,
    validation_is_better,
)


RANDOM_SEED = 42

# The decoder learns ten times faster than the newly unfrozen encoder block.
DECODER_LEARNING_RATE = 1e-4
ENCODER_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUMBER_OF_EPOCHS = 2

MAX_TRAINING_BATCHES = None
MAX_VALIDATION_BATCHES = None
PREDICTION_THRESHOLD = 0.45
RESUME_IF_AVAILABLE = True

CHECKPOINT_DIRECTORY = Path("checkpoints")
SOURCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "balanced_decoder_stage_best.pth"
)
LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "fine_tune_stage_last.pth"
)
BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "fine_tune_stage_best.pth"
)

SOURCE_TRAINING_STAGE = "balanced_weighted_decoder_only"
TRAINING_STAGE = "balanced_weighted_partial_encoder_finetune"


def load_torch_checkpoint(path, device):
    """Load a checkpoint on old and new PyTorch versions."""

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def load_source_checkpoint(model, device):
    """Load the proven decoder-only checkpoint without changing it."""

    if not SOURCE_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "The decoder-only best checkpoint was not found at: "
            f"{SOURCE_CHECKPOINT_PATH.resolve()}"
        )

    checkpoint = load_torch_checkpoint(
        SOURCE_CHECKPOINT_PATH,
        device,
    )

    if checkpoint.get("training_stage") != SOURCE_TRAINING_STAGE:
        raise ValueError(
            "The source checkpoint belongs to an unexpected "
            f"training stage: {checkpoint.get('training_stage')!r}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def configure_partial_fine_tuning(model):
    """Train the decoder and only the deepest ResNet34 encoder block."""

    model.freeze_encoder()

    for parameter in model.encoder_4.parameters():
        parameter.requires_grad = True


def create_optimizer(model):
    """Use a smaller learning rate for pretrained encoder parameters."""

    encoder_parameters = [
        parameter
        for parameter in model.encoder_4.parameters()
        if parameter.requires_grad
    ]
    encoder_parameter_ids = {
        id(parameter) for parameter in encoder_parameters
    }

    decoder_parameters = [
        parameter
        for parameter in model.parameters()
        if (
            parameter.requires_grad
            and id(parameter) not in encoder_parameter_ids
        )
    ]

    if not encoder_parameters or not decoder_parameters:
        raise ValueError(
            "Expected trainable parameters in both the decoder "
            "and encoder_4."
        )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decoder_parameters,
                "lr": DECODER_LEARNING_RATE,
            },
            {
                "params": encoder_parameters,
                "lr": ENCODER_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    return optimizer, decoder_parameters, encoder_parameters


def calculate_dice_scores(logits, targets):
    """Calculate one thresholded Dice score per image."""

    predictions = torch.sigmoid(logits) >= PREDICTION_THRESHOLD
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


def run_one_epoch(
    model,
    data_loader,
    criterion,
    device,
    maximum_batches,
    optimizer=None,
):
    """Run one training or validation epoch."""

    is_training = optimizer is not None

    if is_training:
        model.train()

        # Keep all pretrained BatchNorm statistics fixed. Gradients still
        # flow through encoder_4's trainable convolution and affine weights.
        for encoder_module in model.encoder_modules():
            encoder_module.eval()
    else:
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

    grad_context = (
        torch.enable_grad() if is_training else torch.no_grad()
    )

    with grad_context:
        for batch_number, batch in enumerate(data_loader, start=1):
            if batch_number > total_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            target_masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            output_logits = model(images)
            loss = criterion(output_logits, target_masks)

            if not torch.isfinite(loss):
                raise ValueError(
                    "An epoch produced a non-finite loss."
                )

            if is_training:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                dice_scores = calculate_dice_scores(
                    output_logits,
                    target_masks,
                )
                positive_cases = (
                    target_masks.flatten(start_dim=1).any(dim=1)
                )

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_dice += dice_scores.sum().item()
            total_samples += batch_size
            processed_batches += 1

            if positive_cases.any():
                total_positive_dice += (
                    dice_scores[positive_cases].sum().item()
                )
                total_positive_samples += int(
                    positive_cases.sum().item()
                )

            if should_print_progress(batch_number, total_batches):
                elapsed_minutes = (
                    time.perf_counter() - start_time
                ) / 60.0
                phase = "Training" if is_training else "Validation"

                print(
                    f"  {phase} [{batch_number}/{total_batches}] "
                    f"loss={loss.item():.6f}, "
                    f"dice={dice_scores.mean().item():.6f}, "
                    "positive_dice="
                    f"{batch_positive_dice_text(dice_scores, positive_cases)}, "
                    f"elapsed={elapsed_minutes:.1f} min"
                )

    return create_results(
        total_loss=total_loss,
        total_dice=total_dice,
        total_positive_dice=total_positive_dice,
        total_samples=total_samples,
        total_positive_samples=total_positive_samples,
        processed_batches=processed_batches,
        elapsed_seconds=time.perf_counter() - start_time,
    )


def save_checkpoint(
    path,
    completed_epoch,
    model,
    optimizer,
    baseline_positive_dice,
    best_positive_dice,
    best_loss,
    training_results,
    validation_results,
):
    """Save fine-tuning progress under new checkpoint names."""

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "training_stage": TRAINING_STAGE,
            "source_checkpoint": str(SOURCE_CHECKPOINT_PATH),
            "completed_epoch": completed_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "baseline_validation_positive_dice": (
                baseline_positive_dice
            ),
            "best_validation_positive_dice": best_positive_dice,
            "best_validation_loss": best_loss,
            "training_results": training_results,
            "validation_results": validation_results,
            "configuration": {
                "decoder_learning_rate": DECODER_LEARNING_RATE,
                "encoder_learning_rate": ENCODER_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "positive_pixel_weight": POSITIVE_PIXEL_WEIGHT,
                "prediction_threshold": PREDICTION_THRESHOLD,
                "unfrozen_encoder_block": "encoder_4",
                "encoder_batch_norm_statistics_frozen": True,
            },
        },
        path,
    )


def load_resume_checkpoint(model, optimizer, device):
    """Resume only from this fine-tuning stage."""

    checkpoint = load_torch_checkpoint(LAST_CHECKPOINT_PATH, device)

    if checkpoint.get("training_stage") != TRAINING_STAGE:
        raise ValueError(
            "The resume checkpoint belongs to a different stage."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (
        checkpoint["completed_epoch"] + 1,
        checkpoint["baseline_validation_positive_dice"],
        checkpoint["best_validation_positive_dice"],
        checkpoint["best_validation_loss"],
    )


def count_parameters(parameters):
    return sum(parameter.numel() for parameter in parameters)


def main():
    set_random_seeds(RANDOM_SEED)
    device = choose_device()

    print("Partial encoder fine-tuning stage")
    print("---------------------------------")
    print(f"Device: {device}")
    print(f"Prediction threshold: {PREDICTION_THRESHOLD}")
    print(f"Source checkpoint: {SOURCE_CHECKPOINT_PATH.resolve()}")
    print("Unfrozen encoder block: encoder_4 only")
    print(f"Decoder learning rate: {DECODER_LEARNING_RATE:.1e}")
    print(f"Encoder learning rate: {ENCODER_LEARNING_RATE:.1e}")

    data_loaders = create_dataloaders()
    training_loader = data_loaders["train"]
    validation_loader = find_validation_loader(data_loaders)

    print(f"Training batches: {len(training_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print("The test split will not be used.")

    # No ImageNet download is needed because the source checkpoint contains
    # the complete encoder and decoder state.
    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=False,
        freeze_encoder=True,
    ).to(device)
    configure_partial_fine_tuning(model)

    optimizer, decoder_parameters, encoder_parameters = (
        create_optimizer(model)
    )
    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        positive_pixel_weight=POSITIVE_PIXEL_WEIGHT,
    ).to(device)

    print(
        "Trainable decoder parameters: "
        f"{count_parameters(decoder_parameters):,}"
    )
    print(
        "Trainable encoder_4 parameters: "
        f"{count_parameters(encoder_parameters):,}"
    )

    start_epoch = 1

    if RESUME_IF_AVAILABLE and LAST_CHECKPOINT_PATH.is_file():
        (
            start_epoch,
            baseline_positive_dice,
            best_positive_dice,
            best_loss,
        ) = load_resume_checkpoint(model, optimizer, device)

        print(f"Resumed from: {LAST_CHECKPOINT_PATH.resolve()}")
        print(f"Next epoch: {start_epoch}")
    else:
        source_checkpoint = load_source_checkpoint(model, device)
        print(
            "Loaded decoder-only epoch: "
            f"{source_checkpoint.get('completed_epoch', 'unknown')}"
        )
        print("\nMeasuring the unchanged baseline on validation...")

        baseline_results = run_one_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            maximum_batches=MAX_VALIDATION_BATCHES,
        )
        baseline_positive_dice = baseline_results["positive_dice"]
        best_positive_dice = baseline_positive_dice
        best_loss = baseline_results["loss"]

        print(
            "Baseline validation positive-case Dice: "
            f"{baseline_positive_dice:.6f}"
        )

        # This is a safe fallback copy under a new filename. The original
        # decoder-only checkpoint remains untouched.
        save_checkpoint(
            path=BEST_CHECKPOINT_PATH,
            completed_epoch=0,
            model=model,
            optimizer=optimizer,
            baseline_positive_dice=baseline_positive_dice,
            best_positive_dice=best_positive_dice,
            best_loss=best_loss,
            training_results=None,
            validation_results=baseline_results,
        )

    if start_epoch > NUMBER_OF_EPOCHS:
        print("\nThe requested fine-tuning epochs are complete.")
        return

    active_epoch = start_epoch

    try:
        for epoch_number in range(start_epoch, NUMBER_OF_EPOCHS + 1):
            active_epoch = epoch_number
            print(f"\nEpoch {epoch_number}/{NUMBER_OF_EPOCHS}")

            training_results = run_one_epoch(
                model=model,
                data_loader=training_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                maximum_batches=MAX_TRAINING_BATCHES,
            )
            validation_results = run_one_epoch(
                model=model,
                data_loader=validation_loader,
                criterion=criterion,
                device=device,
                maximum_batches=MAX_VALIDATION_BATCHES,
            )

            print_epoch_summary(training_results, validation_results)

            improved = validation_is_better(
                validation_results,
                best_positive_dice,
                best_loss,
            )

            if improved:
                best_positive_dice = validation_results["positive_dice"]
                best_loss = validation_results["loss"]

            save_checkpoint(
                path=LAST_CHECKPOINT_PATH,
                completed_epoch=epoch_number,
                model=model,
                optimizer=optimizer,
                baseline_positive_dice=baseline_positive_dice,
                best_positive_dice=best_positive_dice,
                best_loss=best_loss,
                training_results=training_results,
                validation_results=validation_results,
            )
            print(f"  Latest checkpoint: {LAST_CHECKPOINT_PATH.resolve()}")

            if improved:
                save_checkpoint(
                    path=BEST_CHECKPOINT_PATH,
                    completed_epoch=epoch_number,
                    model=model,
                    optimizer=optimizer,
                    baseline_positive_dice=baseline_positive_dice,
                    best_positive_dice=best_positive_dice,
                    best_loss=best_loss,
                    training_results=training_results,
                    validation_results=validation_results,
                )
                print(
                    "  New best fine-tuning checkpoint: "
                    f"{BEST_CHECKPOINT_PATH.resolve()}"
                )

    except KeyboardInterrupt:
        save_checkpoint(
            path=LAST_CHECKPOINT_PATH,
            completed_epoch=active_epoch - 1,
            model=model,
            optimizer=optimizer,
            baseline_positive_dice=baseline_positive_dice,
            best_positive_dice=best_positive_dice,
            best_loss=best_loss,
            training_results=None,
            validation_results=None,
        )
        print("\nTraining interrupted safely.")
        print(f"Progress saved to: {LAST_CHECKPOINT_PATH.resolve()}")
        return

    print("\nPartial encoder fine-tuning finished.")
    print(
        "Baseline validation positive-case Dice: "
        f"{baseline_positive_dice:.6f}"
    )
    print(
        "Best validation positive-case Dice: "
        f"{best_positive_dice:.6f}"
    )
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH.resolve()}")


if __name__ == "__main__":
    main()
