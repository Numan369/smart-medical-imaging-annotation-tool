from pathlib import Path
import random
import time

import numpy as np
import torch

from tn3k_dataloaders import create_tn3k_dataloaders
from tn3k_model import (
    TN3KResNet34UNet,
    count_parameters,
)
from tn3k_loss import TN3KV1Loss


# ============================================================
# PATHS
# ============================================================

THYROID_ROOT = (
    Path(__file__).resolve().parent.parent
)


# IMPORTANT:
#
# This is a NEW experiment directory.
# The original tn3k_v1 checkpoints remain untouched.
CHECKPOINT_DIR = (
    THYROID_ROOT
    / "checkpoints"
    / "tn3k_v1_earlystop"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "tn3k_v1_earlystop_best.pth"
)


LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "tn3k_v1_earlystop_last.pth"
)


# ============================================================
# TN3K V1 EARLY-STOPPING CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 512

# Physical GPU batch.
BATCH_SIZE = 2

# Effective batch:
#
# 2 physical samples
# x 2 accumulation steps
# =
# 4 effective samples per normal optimizer update.
GRADIENT_ACCUMULATION_STEPS = 2


# Maximum training duration.
#
# Early stopping may finish before this.
MAX_EPOCHS = 20


LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PREDICTION_THRESHOLD = 0.50

NUM_WORKERS = 0


# ============================================================
# EARLY STOPPING
# ============================================================

# Do not allow early stopping before epoch 6.
MIN_EPOCHS_BEFORE_EARLY_STOPPING = 6


# Stop after 6 consecutive epochs without a meaningful
# validation Dice improvement.
EARLY_STOPPING_PATIENCE = 6


# Validation Dice must improve by at least 0.001
# to reset the early-stopping counter.
#
# Example:
#
# 0.8000 -> 0.8004
# does NOT reset patience.
#
# 0.8000 -> 0.8012
# DOES reset patience.
EARLY_STOPPING_MIN_DELTA = 0.001


# ============================================================
# LR SCHEDULER
# ============================================================

LR_SCHEDULER_PATIENCE = 3

LR_REDUCTION_FACTOR = 0.5

MIN_LEARNING_RATE = 1e-6


# ============================================================
# RESUME
# ============================================================

# If the new early-stopping run gets disconnected,
# continue from its LAST checkpoint.
RESUME_IF_AVAILABLE = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# SEGMENTATION METRICS
# ============================================================

def calculate_batch_metrics(
    logits,
    targets,
    threshold=0.50,
    smooth=1e-6,
):

    probabilities = torch.sigmoid(
        logits
    )


    predictions = (
        probabilities >= threshold
    ).float()


    targets = (
        targets >= 0.5
    ).float()


    # --------------------------------------------------------
    # Flatten per image
    # --------------------------------------------------------

    predictions = predictions.reshape(
        predictions.shape[0],
        -1,
    )


    targets = targets.reshape(
        targets.shape[0],
        -1,
    )


    # --------------------------------------------------------
    # Confusion components
    # --------------------------------------------------------

    true_positive = (
        predictions
        * targets
    ).sum(
        dim=1
    )


    false_positive = (
        predictions
        * (1.0 - targets)
    ).sum(
        dim=1
    )


    false_negative = (
        (1.0 - predictions)
        * targets
    ).sum(
        dim=1
    )


    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice = (
        2.0 * true_positive
        + smooth
    ) / (
        2.0 * true_positive
        + false_positive
        + false_negative
        + smooth
    )


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou = (
        true_positive
        + smooth
    ) / (
        true_positive
        + false_positive
        + false_negative
        + smooth
    )


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        true_positive
        + smooth
    ) / (
        true_positive
        + false_positive
        + smooth
    )


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        true_positive
        + smooth
    ) / (
        true_positive
        + false_negative
        + smooth
    )


    return {

        "dice":
            dice,

        "iou":
            iou,

        "precision":
            precision,

        "recall":
            recall,
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_validation_dice,
    early_stopping_reference_dice,
    epochs_without_meaningful_improvement,
    history,
):

    checkpoint = {

        # ----------------------------------------------------
        # Training progress
        # ----------------------------------------------------

        "epoch":
            epoch,

        "best_validation_dice":
            best_validation_dice,

        "early_stopping_reference_dice":
            early_stopping_reference_dice,

        "epochs_without_meaningful_improvement":
            epochs_without_meaningful_improvement,

        "history":
            history,


        # ----------------------------------------------------
        # Model / optimizer / scheduler
        # ----------------------------------------------------

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),


        # ----------------------------------------------------
        # Experiment configuration
        # ----------------------------------------------------

        "config": {

            "experiment":
                "TN3K V1 Early Stopping",

            "seed":
                SEED,

            "image_size":
                IMAGE_SIZE,

            "batch_size":
                BATCH_SIZE,

            "gradient_accumulation_steps":
                GRADIENT_ACCUMULATION_STEPS,

            "effective_batch_size":
                (
                    BATCH_SIZE
                    * GRADIENT_ACCUMULATION_STEPS
                ),

            "max_epochs":
                MAX_EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "prediction_threshold":
                PREDICTION_THRESHOLD,

            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,

            "early_stopping_min_delta":
                EARLY_STOPPING_MIN_DELTA,

            "minimum_epochs_before_early_stopping":
                MIN_EPOCHS_BEFORE_EARLY_STOPPING,

            "lr_scheduler_patience":
                LR_SCHEDULER_PATIENCE,

            "lr_reduction_factor":
                LR_REDUCTION_FACTOR,

            "architecture":
                "ResNet34 U-Net",

            "encoder_pretraining":
                "ImageNet",

            "loss":
                (
                    "0.50 BCEWithLogits "
                    "+ 0.50 Soft Dice"
                ),

            "optimizer":
                "AdamW",

            "split":
                "TN3K official Fold 0",

            "train_samples":
                2303,

            "validation_samples":
                576,

            "official_test_samples":
                614,

            "official_test_used_during_training":
                False,
        },
    }


    # --------------------------------------------------------
    # Safer save:
    # write temporary file first.
    # --------------------------------------------------------

    temporary_path = Path(
        str(path) + ".tmp"
    )


    torch.save(
        checkpoint,
        temporary_path,
    )


    temporary_path.replace(
        path
    )


# ============================================================
# LOAD CHECKPOINT / RESUME
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device,
):

    print()
    print("=" * 70)
    print("RESUMING TN3K V1 EARLY-STOPPING RUN")
    print("=" * 70)

    print()
    print(
        "Loading:"
    )

    print(
        path
    )


    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )


    # ========================================================
    # BASIC CONFIGURATION COMPATIBILITY CHECK
    # ========================================================

    saved_config = checkpoint.get(
        "config",
        {}
    )


    expected_values = {

        "image_size":
            IMAGE_SIZE,

        "batch_size":
            BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "train_samples":
            2303,

        "validation_samples":
            576,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "early_stopping_min_delta":
            EARLY_STOPPING_MIN_DELTA,
    }


    for key, expected_value in (
        expected_values.items()
    ):

        if key in saved_config:

            saved_value = (
                saved_config[key]
            )


            if saved_value != expected_value:

                raise RuntimeError(
                    "Checkpoint configuration mismatch.\n"
                    f"Field: {key}\n"
                    f"Saved: {saved_value}\n"
                    f"Current: {expected_value}"
                )


    # ========================================================
    # RESTORE STATES
    # ========================================================

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )


    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )


    # ========================================================
    # RESTORE TRAINING PROGRESS
    # ========================================================

    completed_epoch = int(
        checkpoint[
            "epoch"
        ]
    )


    start_epoch = (
        completed_epoch + 1
    )


    best_validation_dice = float(
        checkpoint.get(
            "best_validation_dice",
            -1.0,
        )
    )


    early_stopping_reference_dice = float(
        checkpoint.get(
            "early_stopping_reference_dice",
            -1.0,
        )
    )


    epochs_without_meaningful_improvement = int(
        checkpoint.get(
            "epochs_without_meaningful_improvement",
            0,
        )
    )


    history = checkpoint.get(
        "history",
        [],
    )


    print()
    print(
        f"Completed epoch: "
        f"{completed_epoch}"
    )

    print(
        f"Next epoch: "
        f"{start_epoch}"
    )

    print(
        f"Best validation Dice: "
        f"{best_validation_dice:.6f}"
    )

    print(
        f"Early-stopping reference Dice: "
        f"{early_stopping_reference_dice:.6f}"
    )

    print(
        "Epochs without meaningful improvement: "
        f"{epochs_without_meaningful_improvement}"
    )


    print()
    print(
        "Resume state restored successfully."
    )


    return (
        start_epoch,
        best_validation_dice,
        early_stopping_reference_dice,
        epochs_without_meaningful_improvement,
        history,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()


    # --------------------------------------------------------
    # ResNet encoder BatchNorm running statistics stay fixed.
    # --------------------------------------------------------

    model.freeze_encoder_batchnorm_stats()


    optimizer.zero_grad(
        set_to_none=True
    )


    total_loss = 0.0

    total_bce = 0.0

    total_dice_loss = 0.0

    total_samples = 0


    dice_values = []

    iou_values = []

    precision_values = []

    recall_values = []


    number_of_batches = len(
        loader
    )


    # ========================================================
    # BATCH LOOP
    # ========================================================

    for batch_index, batch in enumerate(
        loader
    ):

        images = (
            batch["image"]
            .to(
                device,
                non_blocking=True,
            )
        )


        masks = (
            batch["mask"]
            .to(
                device,
                non_blocking=True,
            )
        )


        current_batch_size = (
            images.shape[0]
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(
            images
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        (
            loss,
            components,
        ) = criterion(
            logits,
            masks,
        )


        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite training loss detected."
            )


        # ----------------------------------------------------
        # Gradient accumulation
        # ----------------------------------------------------

        scaled_loss = (
            loss
            / GRADIENT_ACCUMULATION_STEPS
        )


        scaled_loss.backward()


        should_step = (

            (
                (batch_index + 1)
                % GRADIENT_ACCUMULATION_STEPS
                == 0
            )

            or

            (
                (batch_index + 1)
                == number_of_batches
            )
        )


        if should_step:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )


            optimizer.step()


            optimizer.zero_grad(
                set_to_none=True
            )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        with torch.no_grad():

            metrics = calculate_batch_metrics(
                logits,
                masks,
                threshold=PREDICTION_THRESHOLD,
            )


        dice_values.extend(
            metrics[
                "dice"
            ]
            .detach()
            .cpu()
            .tolist()
        )


        iou_values.extend(
            metrics[
                "iou"
            ]
            .detach()
            .cpu()
            .tolist()
        )


        precision_values.extend(
            metrics[
                "precision"
            ]
            .detach()
            .cpu()
            .tolist()
        )


        recall_values.extend(
            metrics[
                "recall"
            ]
            .detach()
            .cpu()
            .tolist()
        )


        # ----------------------------------------------------
        # Loss accumulation
        # ----------------------------------------------------

        total_loss += (
            components[
                "total_loss"
            ]
            * current_batch_size
        )


        total_bce += (
            components[
                "bce_loss"
            ]
            * current_batch_size
        )


        total_dice_loss += (
            components[
                "dice_loss"
            ]
            * current_batch_size
        )


        total_samples += (
            current_batch_size
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (

            (
                batch_index + 1
            )
            % 100 == 0

            or

            (
                batch_index + 1
                == number_of_batches
            )

        ):

            print(
                f"    Train batch "
                f"{batch_index + 1}"
                f"/{number_of_batches}"
            )


    return {

        "loss":
            total_loss
            / total_samples,

        "bce_loss":
            total_bce
            / total_samples,

        "dice_loss":
            total_dice_loss
            / total_samples,

        "dice":
            float(
                np.mean(
                    dice_values
                )
            ),

        "iou":
            float(
                np.mean(
                    iou_values
                )
            ),

        "precision":
            float(
                np.mean(
                    precision_values
                )
            ),

        "recall":
            float(
                np.mean(
                    recall_values
                )
            ),
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()


    total_loss = 0.0

    total_bce = 0.0

    total_dice_loss = 0.0

    total_samples = 0


    dice_values = []

    iou_values = []

    precision_values = []

    recall_values = []


    number_of_batches = len(
        loader
    )


    for batch_index, batch in enumerate(
        loader
    ):

        images = (
            batch["image"]
            .to(
                device,
                non_blocking=True,
            )
        )


        masks = (
            batch["mask"]
            .to(
                device,
                non_blocking=True,
            )
        )


        current_batch_size = (
            images.shape[0]
        )


        logits = model(
            images
        )


        (
            loss,
            components,
        ) = criterion(
            logits,
            masks,
        )


        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite validation loss detected."
            )


        metrics = calculate_batch_metrics(
            logits,
            masks,
            threshold=PREDICTION_THRESHOLD,
        )


        dice_values.extend(
            metrics[
                "dice"
            ]
            .cpu()
            .tolist()
        )


        iou_values.extend(
            metrics[
                "iou"
            ]
            .cpu()
            .tolist()
        )


        precision_values.extend(
            metrics[
                "precision"
            ]
            .cpu()
            .tolist()
        )


        recall_values.extend(
            metrics[
                "recall"
            ]
            .cpu()
            .tolist()
        )


        total_loss += (
            components[
                "total_loss"
            ]
            * current_batch_size
        )


        total_bce += (
            components[
                "bce_loss"
            ]
            * current_batch_size
        )


        total_dice_loss += (
            components[
                "dice_loss"
            ]
            * current_batch_size
        )


        total_samples += (
            current_batch_size
        )


        if (

            (
                batch_index + 1
            )
            % 100 == 0

            or

            (
                batch_index + 1
                == number_of_batches
            )

        ):

            print(
                f"    Validation batch "
                f"{batch_index + 1}"
                f"/{number_of_batches}"
            )


    return {

        "loss":
            total_loss
            / total_samples,

        "bce_loss":
            total_bce
            / total_samples,

        "dice_loss":
            total_dice_loss
            / total_samples,

        "dice":
            float(
                np.mean(
                    dice_values
                )
            ),

        "iou":
            float(
                np.mean(
                    iou_values
                )
            ),

        "precision":
            float(
                np.mean(
                    precision_values
                )
            ),

        "recall":
            float(
                np.mean(
                    recall_values
                )
            ),
    }


# ============================================================
# PRINT EPOCH RESULTS
# ============================================================

def print_epoch_results(
    epoch,
    train_metrics,
    validation_metrics,
    learning_rate,
    elapsed_seconds,
):

    print()
    print("=" * 70)

    print(
        f"EPOCH "
        f"{epoch}/{MAX_EPOCHS}"
    )

    print("=" * 70)


    print()
    print("TRAIN")

    print(
        f"  Loss:      "
        f"{train_metrics['loss']:.6f}"
    )

    print(
        f"  BCE:       "
        f"{train_metrics['bce_loss']:.6f}"
    )

    print(
        f"  Dice loss: "
        f"{train_metrics['dice_loss']:.6f}"
    )

    print(
        f"  Dice:      "
        f"{train_metrics['dice']:.6f}"
    )

    print(
        f"  IoU:       "
        f"{train_metrics['iou']:.6f}"
    )

    print(
        f"  Precision: "
        f"{train_metrics['precision']:.6f}"
    )

    print(
        f"  Recall:    "
        f"{train_metrics['recall']:.6f}"
    )


    print()
    print("VALIDATION")

    print(
        f"  Loss:      "
        f"{validation_metrics['loss']:.6f}"
    )

    print(
        f"  BCE:       "
        f"{validation_metrics['bce_loss']:.6f}"
    )

    print(
        f"  Dice loss: "
        f"{validation_metrics['dice_loss']:.6f}"
    )

    print(
        f"  Dice:      "
        f"{validation_metrics['dice']:.6f}"
    )

    print(
        f"  IoU:       "
        f"{validation_metrics['iou']:.6f}"
    )

    print(
        f"  Precision: "
        f"{validation_metrics['precision']:.6f}"
    )

    print(
        f"  Recall:    "
        f"{validation_metrics['recall']:.6f}"
    )


    print()
    print(
        f"Learning rate: "
        f"{learning_rate:.8f}"
    )


    print(
        f"Epoch time: "
        f"{elapsed_seconds / 60:.2f} minutes"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TN3K V1 EARLY-STOPPING TRAINING")
    print("=" * 70)


    # ========================================================
    # SEED
    # ========================================================

    set_seed(
        SEED
    )


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )


    print()
    print(
        f"Device: "
        f"{device}"
    )


    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


    # ========================================================
    # CONFIGURATION
    # ========================================================

    print()
    print("CONFIGURATION")
    print("-" * 70)


    print(
        f"Image size: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )


    print(
        f"Physical batch size: "
        f"{BATCH_SIZE}"
    )


    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )


    print(
        f"Effective batch size: "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )


    print(
        f"Maximum epochs: "
        f"{MAX_EPOCHS}"
    )


    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )


    print(
        f"Early-stop patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )


    print(
        f"Early-stop min delta: "
        f"{EARLY_STOPPING_MIN_DELTA}"
    )


    print(
        f"Minimum epochs before stop: "
        f"{MIN_EPOCHS_BEFORE_EARLY_STOPPING}"
    )


    print(
        f"LR scheduler patience: "
        f"{LR_SCHEDULER_PATIENCE}"
    )


    print(
        f"Automatic resume: "
        f"{RESUME_IF_AVAILABLE}"
    )


    # ========================================================
    # DATA
    # ========================================================

    print()
    print(
        "Creating TN3K Fold-0 DataLoaders..."
    )


    (
        train_loader,
        validation_loader,
    ) = create_tn3k_dataloaders(

        batch_size=BATCH_SIZE,

        num_workers=NUM_WORKERS,
    )


    print()
    print(
        f"Train samples: "
        f"{len(train_loader.dataset)}"
    )


    print(
        f"Validation samples: "
        f"{len(validation_loader.dataset)}"
    )


    if len(train_loader.dataset) != 2303:

        raise RuntimeError(
            "Expected 2303 training images."
        )


    if len(validation_loader.dataset) != 576:

        raise RuntimeError(
            "Expected 576 validation images."
        )


    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Creating ResNet34 U-Net..."
    )


    model = TN3KResNet34UNet(
        use_pretrained_encoder=True,
    )


    model.configure_v1_trainable_layers()


    model = model.to(
        device
    )


    (
        total_parameters,
        trainable_parameters,
        frozen_parameters,
    ) = count_parameters(
        model
    )


    print()
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )


    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )


    print(
        f"Frozen parameters: "
        f"{frozen_parameters:,}"
    )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = TN3KV1Loss(

        bce_weight=0.50,

        dice_weight=0.50,
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(

        filter(
            lambda parameter:
                parameter.requires_grad,

            model.parameters(),
        ),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )


    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=LR_REDUCTION_FACTOR,

            patience=LR_SCHEDULER_PATIENCE,

            min_lr=MIN_LEARNING_RATE,
        )
    )


    # ========================================================
    # FRESH TRAINING STATE
    # ========================================================

    start_epoch = 1

    best_validation_dice = -1.0

    early_stopping_reference_dice = -1.0

    epochs_without_meaningful_improvement = 0

    history = []


    # ========================================================
    # RESUME IF THIS NEW RUN WAS INTERRUPTED
    # ========================================================

    if (

        RESUME_IF_AVAILABLE

        and

        LAST_CHECKPOINT.exists()

    ):

        (
            start_epoch,
            best_validation_dice,
            early_stopping_reference_dice,
            epochs_without_meaningful_improvement,
            history,
        ) = load_checkpoint(

            path=LAST_CHECKPOINT,

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            device=device,
        )


    else:

        print()
        print(
            "No early-stopping checkpoint found."
        )

        print(
            "Starting NEW training from epoch 1."
        )


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    stopped_early = False

    stop_epoch = None


    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):

        epoch_start = (
            time.time()
        )


        print()
        print("#" * 70)

        print(
            f"STARTING EPOCH "
            f"{epoch}/{MAX_EPOCHS}"
        )

        print("#" * 70)


        # ====================================================
        # TRAIN
        # ====================================================

        train_metrics = train_one_epoch(

            model=model,

            loader=train_loader,

            criterion=criterion,

            optimizer=optimizer,

            device=device,
        )


        # ====================================================
        # VALIDATE
        # ====================================================

        validation_metrics = validate(

            model=model,

            loader=validation_loader,

            criterion=criterion,

            device=device,
        )


        current_validation_dice = (
            validation_metrics[
                "dice"
            ]
        )


        # ====================================================
        # LR SCHEDULER
        # ====================================================

        scheduler.step(
            current_validation_dice
        )


        current_learning_rate = (
            optimizer
            .param_groups[0]["lr"]
        )


        epoch_elapsed = (
            time.time()
            - epoch_start
        )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print_epoch_results(

            epoch=epoch,

            train_metrics=train_metrics,

            validation_metrics=(
                validation_metrics
            ),

            learning_rate=(
                current_learning_rate
            ),

            elapsed_seconds=(
                epoch_elapsed
            ),
        )


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        epoch_record = {

            "epoch":
                epoch,

            "train":
                train_metrics,

            "validation":
                validation_metrics,

            "learning_rate":
                current_learning_rate,

            "epoch_seconds":
                epoch_elapsed,
        }


        history.append(
            epoch_record
        )


        # ====================================================
        # BEST CHECKPOINT
        #
        # Any actual improvement becomes best.pth.
        # ====================================================

        if (
            current_validation_dice
            > best_validation_dice
        ):

            best_validation_dice = (
                current_validation_dice
            )


            print()
            print(
                "*** NEW BEST VALIDATION MODEL ***"
            )


            print(
                f"Best validation Dice: "
                f"{best_validation_dice:.6f}"
            )


            # Save after early-stopping state is updated below.


        # ====================================================
        # EARLY-STOPPING LOGIC
        #
        # This is intentionally separate from best checkpoint.
        #
        # A tiny increase can still become best.pth,
        # but must exceed MIN_DELTA to reset patience.
        # ====================================================

        meaningful_improvement = (

            current_validation_dice
            >=
            (
                early_stopping_reference_dice
                + EARLY_STOPPING_MIN_DELTA
            )
        )


        if meaningful_improvement:

            early_stopping_reference_dice = (
                current_validation_dice
            )


            epochs_without_meaningful_improvement = 0


            print()
            print(
                "EARLY STOPPING:"
            )

            print(
                "  Meaningful validation improvement."
            )

            print(
                "  Patience counter reset to 0."
            )


        else:

            epochs_without_meaningful_improvement += 1


            print()
            print(
                "EARLY STOPPING:"
            )

            print(
                "  No meaningful validation improvement."
            )

            print(
                f"  Counter: "
                f"{epochs_without_meaningful_improvement}"
                f"/{EARLY_STOPPING_PATIENCE}"
            )


            print(
                f"  Reference Dice: "
                f"{early_stopping_reference_dice:.6f}"
            )


            print(
                f"  Current Dice:   "
                f"{current_validation_dice:.6f}"
            )


        # ====================================================
        # SAVE BEST CHECKPOINT
        #
        # Save if this epoch equals current overall best.
        # ====================================================

        if (
            current_validation_dice
            == best_validation_dice
        ):

            save_checkpoint(

                path=BEST_CHECKPOINT,

                model=model,

                optimizer=optimizer,

                scheduler=scheduler,

                epoch=epoch,

                best_validation_dice=(
                    best_validation_dice
                ),

                early_stopping_reference_dice=(
                    early_stopping_reference_dice
                ),

                epochs_without_meaningful_improvement=(
                    epochs_without_meaningful_improvement
                ),

                history=history,
            )


            print()
            print(
                "Saved best checkpoint:"
            )

            print(
                BEST_CHECKPOINT
            )


        # ====================================================
        # ALWAYS SAVE LAST
        # ====================================================

        save_checkpoint(

            path=LAST_CHECKPOINT,

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            epoch=epoch,

            best_validation_dice=(
                best_validation_dice
            ),

            early_stopping_reference_dice=(
                early_stopping_reference_dice
            ),

            epochs_without_meaningful_improvement=(
                epochs_without_meaningful_improvement
            ),

            history=history,
        )


        print()
        print(
            "Saved resumable last checkpoint:"
        )

        print(
            LAST_CHECKPOINT
        )


        # ====================================================
        # STATUS
        # ====================================================

        print()
        print(
            f"Best validation Dice so far: "
            f"{best_validation_dice:.6f}"
        )


        print(
            "Epochs without meaningful improvement: "
            f"{epochs_without_meaningful_improvement}"
            f"/{EARLY_STOPPING_PATIENCE}"
        )


        # ====================================================
        # EARLY STOP CONDITION
        # ====================================================

        enough_epochs_completed = (

            epoch
            >=
            MIN_EPOCHS_BEFORE_EARLY_STOPPING
        )


        patience_exhausted = (

            epochs_without_meaningful_improvement
            >=
            EARLY_STOPPING_PATIENCE
        )


        if (

            enough_epochs_completed

            and

            patience_exhausted

        ):

            stopped_early = True

            stop_epoch = epoch


            print()
            print("=" * 70)
            print("EARLY STOPPING TRIGGERED")
            print("=" * 70)


            print()
            print(
                f"Stopped after epoch: "
                f"{epoch}"
            )


            print(
                "Reason:"
            )

            print(
                f"No meaningful validation Dice "
                f"improvement of at least "
                f"{EARLY_STOPPING_MIN_DELTA:.4f} "
                f"for "
                f"{EARLY_STOPPING_PATIENCE} "
                f"consecutive epochs."
            )


            print()
            print(
                f"Best validation Dice: "
                f"{best_validation_dice:.6f}"
            )


            print()
            print(
                "Best model remains safely stored at:"
            )


            print(
                BEST_CHECKPOINT
            )


            break


    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TN3K V1 EARLY-STOPPING RUN FINISHED"
    )
    print("=" * 70)


    print()
    print(
        f"Best validation Dice: "
        f"{best_validation_dice:.6f}"
    )


    if stopped_early:

        print(
            f"Stopped early at epoch: "
            f"{stop_epoch}"
        )

    else:

        print(
            f"Completed maximum epochs: "
            f"{MAX_EPOCHS}"
        )


    print()
    print(
        "Best checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )


    print()
    print(
        "Last/resume checkpoint:"
    )

    print(
        LAST_CHECKPOINT
    )


    print()
    print(
        "Official TN3K test set was NOT evaluated."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()