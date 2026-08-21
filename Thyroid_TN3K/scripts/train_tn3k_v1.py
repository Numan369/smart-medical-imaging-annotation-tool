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

CHECKPOINT_DIR = (
    THYROID_ROOT
    / "checkpoints"
    / "tn3k_v1"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "tn3k_v1_best.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "tn3k_v1_last.pth"
)


# ============================================================
# TN3K V1 CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 512

# Physical number of images loaded by the GPU at once.
BATCH_SIZE = 2

# Two physical batches are accumulated before optimizer.step().
#
# Effective batch:
#
# 2 physical batch
# x
# 2 accumulation steps
# =
# 4 effective samples per normal optimizer update.
GRADIENT_ACCUMULATION_STEPS = 2

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

# Initial/fixed V1 threshold used for validation metrics.
PREDICTION_THRESHOLD = 0.50

# Conservative setting for maximum reliability.
NUM_WORKERS = 0

# If tn3k_v1_last.pth exists, automatically resume.
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
    """
    Calculate binary segmentation metrics independently
    for every image in the batch.

    Returns per-image:

        Dice
        IoU
        Precision
        Recall
    """

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities
        >= threshold
    ).float()

    targets = (
        targets
        >= 0.5
    ).float()


    # --------------------------------------------------------
    # Flatten every image independently
    #
    # [B,1,H,W]
    # ->
    # [B,pixels]
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
    #
    # 2TP
    # -----------------
    # 2TP + FP + FN
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
    # Intersection over Union
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
    #
    # Of predicted nodule pixels,
    # how many were actually nodule?
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
    # Recall / sensitivity
    #
    # Of actual nodule pixels,
    # how many did we find?
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
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_validation_dice,
    history,
):
    """
    Save everything needed to resume training.

    A temporary file is written first. Only after torch.save()
    completes successfully is it moved to the actual checkpoint
    path.

    This reduces the chance of replacing a good checkpoint with
    an incomplete file if saving is interrupted.
    """

    checkpoint = {

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        "epoch":
            epoch,

        "best_validation_dice":
            best_validation_dice,

        "history":
            history,


        # ----------------------------------------------------
        # Neural network
        # ----------------------------------------------------

        "model_state_dict":
            model.state_dict(),


        # ----------------------------------------------------
        # Optimizer
        #
        # Important for AdamW because it contains optimizer
        # state/momentum information.
        # ----------------------------------------------------

        "optimizer_state_dict":
            optimizer.state_dict(),


        # ----------------------------------------------------
        # LR scheduler
        # ----------------------------------------------------

        "scheduler_state_dict":
            scheduler.state_dict(),


        # ----------------------------------------------------
        # Experiment configuration
        # ----------------------------------------------------

        "config": {

            "experiment":
                "TN3K V1",

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

            "num_epochs":
                NUM_EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "prediction_threshold":
                PREDICTION_THRESHOLD,

            "architecture":
                "ResNet34 U-Net",

            "encoder_pretraining":
                "ImageNet",

            "normalization":
                (
                    "Encoder BatchNorm running statistics "
                    "fixed; decoder GroupNorm"
                ),

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
    # Temporary checkpoint
    # --------------------------------------------------------

    temporary_path = Path(
        str(path) + ".tmp"
    )


    torch.save(
        checkpoint,
        temporary_path,
    )


    # --------------------------------------------------------
    # Replace actual checkpoint only after successful save.
    # --------------------------------------------------------

    temporary_path.replace(
        path
    )


# ============================================================
# CHECKPOINT LOAD / AUTOMATIC RESUME
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device,
):
    """
    Restore:

        model
        optimizer
        scheduler
        completed epoch
        best validation Dice
        history

    Returns the NEXT epoch to execute.
    """

    print()
    print("=" * 70)
    print("RESUMING TN3K V1 TRAINING")
    print("=" * 70)

    print()
    print(
        "Checkpoint:"
    )

    print(
        path
    )


    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )


    # ========================================================
    # CONFIGURATION COMPATIBILITY CHECK
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
    }


    for key, expected_value in (
        expected_values.items()
    ):

        # Older checkpoints could theoretically lack
        # a field. Only compare when it exists.
        if key in saved_config:

            saved_value = (
                saved_config[key]
            )

            if (
                saved_value
                != expected_value
            ):

                raise RuntimeError(
                    "Checkpoint configuration "
                    "does not match current training.\n"
                    f"Field: {key}\n"
                    f"Saved: {saved_value}\n"
                    f"Current: {expected_value}"
                )


    # ========================================================
    # RESTORE MODEL
    # ========================================================

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    # ========================================================
    # RESTORE OPTIMIZER
    # ========================================================

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )


    # ========================================================
    # RESTORE SCHEDULER
    # ========================================================

    if (
        "scheduler_state_dict"
        in checkpoint
    ):

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    else:

        print()
        print(
            "WARNING:"
        )

        print(
            "Checkpoint does not contain "
            "scheduler state."
        )


    # ========================================================
    # RESTORE PROGRESS
    # ========================================================

    completed_epoch = int(
        checkpoint[
            "epoch"
        ]
    )


    start_epoch = (
        completed_epoch
        + 1
    )


    best_validation_dice = float(
        checkpoint.get(
            "best_validation_dice",
            -1.0,
        )
    )


    history = checkpoint.get(
        "history",
        [],
    )


    # ========================================================
    # PRINT
    # ========================================================

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
        f"Best validation Dice so far: "
        f"{best_validation_dice:.6f}"
    )

    print(
        f"History records: "
        f"{len(history)}"
    )


    if saved_config:

        print()
        print(
            "Saved experiment:"
        )

        print(
            saved_config.get(
                "experiment",
                "Unknown",
            )
        )


    print()
    print(
        "Checkpoint resume: READY"
    )

    print("=" * 70)


    return (
        start_epoch,
        best_validation_dice,
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
    """
    Train the network for one complete pass through
    the 2303 Fold-0 training images.
    """

    model.train()


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # model.train() puts all BatchNorm modules back into
    # training mode.
    #
    # Physical batch size is only 2, therefore immediately
    # restore encoder BatchNorm modules to eval mode so their
    # running mean/variance remain stable.
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
    # TRAINING BATCH LOOP
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


        # ====================================================
        # FORWARD
        # ====================================================

        logits = model(
            images
        )


        # ====================================================
        # LOSS
        # ====================================================

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


        # ====================================================
        # GRADIENT ACCUMULATION
        # ====================================================

        scaled_loss = (
            loss
            / GRADIENT_ACCUMULATION_STEPS
        )


        scaled_loss.backward()


        # ----------------------------------------------------
        # Normally update after every two physical batches.
        #
        # Also update on the final batch of the epoch.
        # ----------------------------------------------------

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

            # ------------------------------------------------
            # Safety against extreme gradients.
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )


            optimizer.step()


            optimizer.zero_grad(
                set_to_none=True
            )


        # ====================================================
        # TRAINING METRICS
        # ====================================================

        with torch.no_grad():

            metrics = (
                calculate_batch_metrics(
                    logits,
                    masks,
                    threshold=(
                        PREDICTION_THRESHOLD
                    ),
                )
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


        # ====================================================
        # LOSS ACCUMULATION
        # ====================================================

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


        # ====================================================
        # PROGRESS
        # ====================================================

        if (

            (
                batch_index + 1
            )
            % 100
            == 0

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


    # ========================================================
    # EPOCH AVERAGES
    # ========================================================

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
    """
    Evaluate all 576 Fold-0 validation images.

    No random augmentation.
    No gradient calculation.
    No optimizer update.
    """

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


    # ========================================================
    # VALIDATION LOOP
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


        # ====================================================
        # FORWARD
        # ====================================================

        logits = model(
            images
        )


        # ====================================================
        # LOSS
        # ====================================================

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


        # ====================================================
        # METRICS
        # ====================================================

        metrics = (
            calculate_batch_metrics(
                logits,
                masks,
                threshold=(
                    PREDICTION_THRESHOLD
                ),
            )
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


        # ====================================================
        # ACCUMULATE LOSS
        # ====================================================

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


        # ====================================================
        # PROGRESS
        # ====================================================

        if (

            (
                batch_index + 1
            )
            % 100
            == 0

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


    # ========================================================
    # VALIDATION AVERAGES
    # ========================================================

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
        f"{epoch}/{NUM_EPOCHS}"
    )

    print("=" * 70)


    # ========================================================
    # TRAIN
    # ========================================================

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


    # ========================================================
    # VALIDATION
    # ========================================================

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


    # ========================================================
    # TRAINING STATE
    # ========================================================

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
    print("TN3K V1 TRAINING")
    print("=" * 70)


    # ========================================================
    # RANDOM SEED
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


    if (
        device.type
        == "cuda"
    ):

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
        f"Epochs: "
        f"{NUM_EPOCHS}"
    )


    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )


    print(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )


    print(
        f"Prediction threshold: "
        f"{PREDICTION_THRESHOLD}"
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


    # ========================================================
    # DATASET SAFETY
    # ========================================================

    if (
        len(train_loader.dataset)
        != 2303
    ):

        raise RuntimeError(
            "Unexpected TN3K training count. "
            f"Expected 2303, found "
            f"{len(train_loader.dataset)}"
        )


    if (
        len(validation_loader.dataset)
        != 576
    ):

        raise RuntimeError(
            "Unexpected TN3K validation count. "
            f"Expected 576, found "
            f"{len(validation_loader.dataset)}"
        )


    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Creating TN3K ResNet34 U-Net..."
    )


    model = TN3KResNet34UNet(
        use_pretrained_encoder=True,
    )


    # Frozen:
    #
    # stem
    # encoder1
    # encoder2
    #
    # Trainable:
    #
    # encoder3
    # encoder4
    # decoder
    model.configure_v1_trainable_layers()


    model = model.to(
        device
    )


    # ========================================================
    # PARAMETER COUNTS
    # ========================================================

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
    # LEARNING-RATE SCHEDULER
    #
    # If validation Dice fails to improve for several epochs,
    # reduce the learning rate.
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=3,

            min_lr=1e-6,
        )
    )


    # ========================================================
    # INITIAL TRAINING STATE
    # ========================================================

    history = []

    best_validation_dice = -1.0

    start_epoch = 1


    # ========================================================
    # AUTOMATIC RESUME
    # ========================================================

    if (
        RESUME_IF_AVAILABLE

        and

        LAST_CHECKPOINT.exists()
    ):

        (
            start_epoch,
            best_validation_dice,
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
            "No resumable checkpoint found."
        )

        print(
            "Starting TN3K V1 from epoch 1."
        )


    # ========================================================
    # ALREADY COMPLETE?
    # ========================================================

    if (
        start_epoch
        > NUM_EPOCHS
    ):

        print()
        print("=" * 70)

        print(
            "TN3K V1 IS ALREADY COMPLETE"
        )

        print("=" * 70)

        print()

        print(
            f"Checkpoint has already completed "
            f"epoch {start_epoch - 1}."
        )

        print(
            f"Configured training length is "
            f"{NUM_EPOCHS} epochs."
        )

        print()

        print(
            f"Best validation Dice: "
            f"{best_validation_dice:.6f}"
        )

        print()

        print(
            "No additional training was performed."
        )

        return


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        start_epoch,
        NUM_EPOCHS + 1,
    ):

        epoch_start = (
            time.time()
        )


        print()
        print("#" * 70)

        print(
            f"STARTING EPOCH "
            f"{epoch}/{NUM_EPOCHS}"
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


        # ====================================================
        # LEARNING-RATE SCHEDULER
        # ====================================================

        scheduler.step(
            validation_metrics[
                "dice"
            ]
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

            train_metrics=(
                train_metrics
            ),

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
        # ADD TO HISTORY
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
        # BEST MODEL CHECK
        # ====================================================

        current_validation_dice = (
            validation_metrics[
                "dice"
            ]
        )


        is_new_best = (
            current_validation_dice
            > best_validation_dice
        )


        if is_new_best:

            best_validation_dice = (
                current_validation_dice
            )


            # ------------------------------------------------
            # Save BEST checkpoint.
            # ------------------------------------------------

            save_checkpoint(

                path=BEST_CHECKPOINT,

                model=model,

                optimizer=optimizer,

                scheduler=scheduler,

                epoch=epoch,

                best_validation_dice=(
                    best_validation_dice
                ),

                history=history,
            )


            print()
            print(
                "*** NEW BEST MODEL ***"
            )


            print(
                f"Best validation Dice: "
                f"{best_validation_dice:.6f}"
            )


            print(
                "Saved best checkpoint:"
            )


            print(
                BEST_CHECKPOINT
            )


        # ====================================================
        # ALWAYS SAVE LAST CHECKPOINT
        #
        # This happens after best_validation_dice is updated,
        # so last.pth always stores the true current best.
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
        # GPU MEMORY REPORT
        # ====================================================

        if (
            device.type
            == "cuda"
        ):

            allocated_gb = (
                torch.cuda.memory_allocated()
                / 1024**3
            )


            reserved_gb = (
                torch.cuda.memory_reserved()
                / 1024**3
            )


            print()
            print(
                f"GPU allocated: "
                f"{allocated_gb:.2f} GB"
            )


            print(
                f"GPU reserved:  "
                f"{reserved_gb:.2f} GB"
            )


        # ====================================================
        # CURRENT BEST
        # ====================================================

        print()
        print(
            f"Best validation Dice so far: "
            f"{best_validation_dice:.6f}"
        )


    # ========================================================
    # TRAINING FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TN3K V1 TRAINING COMPLETE"
    )
    print("=" * 70)


    print()
    print(
        f"Best validation Dice: "
        f"{best_validation_dice:.6f}"
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
        "Last checkpoint:"
    )

    print(
        LAST_CHECKPOINT
    )


    print()
    print(
        "Official TN3K test set has NOT "
        "been evaluated."
    )


    print()
    print(
        "Do not evaluate the official test set "
        "until model-development decisions are complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()