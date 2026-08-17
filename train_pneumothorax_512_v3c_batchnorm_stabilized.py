"""Train Pneumothorax Model V3C with stable BatchNorm statistics.

V3C repeats the V3B small-lesion sampling experiment from the locked V1
epoch-4 checkpoint. The only controlled change relative to V3B is that all
BatchNorm layers use their saved running statistics during training instead of
estimating mean and variance from batches of only two images.

BatchNorm affine parameters are not additionally frozen: gamma and beta remain
trainable wherever the inherited fine-tuning phase already permits them. Loss,
sampling, augmentation, optimizer, learning rates, trainable phases,
resolution, prediction threshold, validation and early stopping are unchanged.
The test split is never created or accessed.
"""

import math
from pathlib import Path

import torch
from torch import nn

import train_pneumothorax_512_negative_aware as training
import train_pneumothorax_512_v3b_small_lesion_sampling as v3b


EXPECTED_SOURCE_STAGE = "pneumothorax_512_negative_aware_finetune"
EXPECTED_SOURCE_EPOCH = 4
EXPECTED_IMAGE_SIZE = 512
EXPECTED_THRESHOLD = 0.35

CONTROLLED_CHANGE = (
    "freeze BatchNorm running statistics during V3B training"
)
BATCHNORM_MODE = "saved_running_statistics_during_training"

DRIVE_PROJECT_DIRECTORY = Path(
    "/content/drive/MyDrive/SmartMedicalImagingColab"
)
SOURCE_CHECKPOINT_PATH = (
    DRIVE_PROJECT_DIRECTORY
    / "checkpoints"
    / "pneumothorax_512_negative_aware_best.pth"
)

# Keep True for the first two-batch Colab smoke test. After it passes, change
# only this value to False and rerun the script for full V3C training.
SMOKE_TEST = True


ORIGINAL_RUN_EPOCH = training.run_epoch


def almost_equal(first, second):
    return math.isclose(
        float(first),
        float(second),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def batchnorm_modules(model):
    """Return every BatchNorm module in the current architecture."""

    return [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]


def snapshot_batchnorm_statistics(modules):
    """Copy every running-statistic buffer for an invariance check."""

    snapshots = []
    for module in modules:
        snapshots.append(
            {
                "running_mean": (
                    None
                    if module.running_mean is None
                    else module.running_mean.detach().clone()
                ),
                "running_var": (
                    None
                    if module.running_var is None
                    else module.running_var.detach().clone()
                ),
                "num_batches_tracked": (
                    None
                    if module.num_batches_tracked is None
                    else module.num_batches_tracked.detach().clone()
                ),
            }
        )
    return snapshots


def assert_batchnorm_statistics_unchanged(modules, snapshots):
    """Fail immediately if training modified a BatchNorm running buffer."""

    if len(modules) != len(snapshots):
        raise RuntimeError("BatchNorm module count changed during an epoch.")

    for module_number, (module, before) in enumerate(
        zip(modules, snapshots),
        start=1,
    ):
        for buffer_name in (
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ):
            after = getattr(module, buffer_name)
            previous = before[buffer_name]
            if previous is None and after is None:
                continue
            if previous is None or after is None:
                raise RuntimeError(
                    f"BatchNorm module {module_number} changed "
                    f"{buffer_name} availability."
                )
            if not torch.equal(previous, after.detach()):
                raise RuntimeError(
                    f"BatchNorm module {module_number} changed "
                    f"{buffer_name}; refusing an uncontrolled V3C run."
                )


def run_epoch_with_stable_batchnorm(
    model,
    data_loader,
    criterion,
    device,
    maximum_batches,
    optimizer=None,
    scaler=None,
):
    """Run the inherited epoch while freezing BN running statistics."""

    if optimizer is None:
        return ORIGINAL_RUN_EPOCH(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            device=device,
            maximum_batches=maximum_batches,
            optimizer=optimizer,
            scaler=scaler,
        )

    modules = batchnorm_modules(model)
    if not modules:
        raise RuntimeError("No BatchNorm modules were found in the model.")

    snapshots = snapshot_batchnorm_statistics(modules)
    hook_was_applied = {"value": False}

    def use_saved_statistics_before_forward(root_module, inputs):
        del root_module, inputs
        for module in modules:
            module.eval()
        hook_was_applied["value"] = True

    hook = model.register_forward_pre_hook(
        use_saved_statistics_before_forward
    )
    try:
        results = ORIGINAL_RUN_EPOCH(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            device=device,
            maximum_batches=maximum_batches,
            optimizer=optimizer,
            scaler=scaler,
        )
    finally:
        hook.remove()

    if not hook_was_applied["value"]:
        raise RuntimeError("The BatchNorm stabilization hook was not used.")

    assert_batchnorm_statistics_unchanged(modules, snapshots)
    print(
        "  BatchNorm verification: "
        f"{len(modules)} layers kept saved running statistics"
    )
    return results


def save_v3c_checkpoint(
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
    """Save V3C without overwriting any earlier experiment."""

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "training_stage": training.TRAINING_STAGE,
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
                "image_size": training.IMAGE_SIZE,
                "batch_size": training.BATCH_SIZE,
                "gradient_accumulation_steps": (
                    training.GRADIENT_ACCUMULATION_STEPS
                ),
                "prediction_threshold": training.PREDICTION_THRESHOLD,
                "maximum_epochs": training.MAX_EPOCHS,
                "early_stopping_patience": (
                    training.EARLY_STOPPING_PATIENCE
                ),
                "training_positive_fraction": 0.35,
                "tiny_small_positive_sample_fraction": (
                    v3b.TINY_SMALL_SAMPLE_FRACTION
                ),
                "medium_large_positive_sample_fraction": (
                    v3b.MEDIUM_LARGE_SAMPLE_FRACTION
                ),
                "negative_sample_fraction": (
                    v3b.NEGATIVE_SAMPLE_FRACTION
                ),
                "small_lesion_maximum_fraction": (
                    v3b.SMALL_LESION_MAXIMUM_FRACTION
                ),
                "tiny_small_positive_images": (
                    v3b.EXPECTED_TINY_SMALL_POSITIVES
                ),
                "medium_large_positive_images": (
                    v3b.EXPECTED_MEDIUM_LARGE_POSITIVES
                ),
                "negative_images": v3b.EXPECTED_NEGATIVES,
                "source_checkpoint": str(SOURCE_CHECKPOINT_PATH),
                "source_training_stage": EXPECTED_SOURCE_STAGE,
                "source_checkpoint_epoch": EXPECTED_SOURCE_EPOCH,
                "comparison_experiment": (
                    "pneumothorax_512_v3b_small_lesion_sampling_finetune"
                ),
                "controlled_change": CONTROLLED_CHANGE,
                "batchnorm_mode": BATCHNORM_MODE,
                "batchnorm_running_statistics_frozen": True,
                "batchnorm_affine_parameters_additionally_frozen": False,
                "augmentation": "paired_conservative_training_only",
                "loss": (
                    "0.45 weighted BCE + 0.35 symmetric positive "
                    "focal Tversky + 0.20 negative-only BCE"
                ),
                "bce_weight": training.BCE_WEIGHT,
                "positive_tversky_weight": (
                    training.POSITIVE_TVERSKY_WEIGHT
                ),
                "negative_bce_weight": training.NEGATIVE_BCE_WEIGHT,
                "positive_pixel_weight": training.POSITIVE_PIXEL_WEIGHT,
                "tversky_false_positive_weight": (
                    training.TVERSKY_FALSE_POSITIVE_WEIGHT
                ),
                "tversky_false_negative_weight": (
                    training.TVERSKY_FALSE_NEGATIVE_WEIGHT
                ),
                "checkpoint_selection": (
                    "harmonic mean of validation positive Dice and "
                    "negative empty-mask accuracy"
                ),
                "validation_split_used": True,
                "test_split_used": False,
            },
        },
        path,
    )


def load_v3c_resume_state(model, optimizer, scheduler, scaler, device):
    """Resume only a checkpoint created by this exact V3C experiment."""

    checkpoint = training.load_torch_checkpoint(
        training.LAST_CHECKPOINT_PATH,
        device,
    )

    if checkpoint.get("training_stage") != training.TRAINING_STAGE:
        raise ValueError("Resume checkpoint belongs to another stage.")

    configuration = checkpoint.get("configuration", {})
    required = {
        "controlled_change": CONTROLLED_CHANGE,
        "batchnorm_mode": BATCHNORM_MODE,
        "batchnorm_running_statistics_frozen": True,
        "batchnorm_affine_parameters_additionally_frozen": False,
        "source_training_stage": EXPECTED_SOURCE_STAGE,
        "source_checkpoint_epoch": EXPECTED_SOURCE_EPOCH,
        "tiny_small_positive_sample_fraction": (
            v3b.TINY_SMALL_SAMPLE_FRACTION
        ),
        "medium_large_positive_sample_fraction": (
            v3b.MEDIUM_LARGE_SAMPLE_FRACTION
        ),
        "negative_sample_fraction": v3b.NEGATIVE_SAMPLE_FRACTION,
        "small_lesion_maximum_fraction": (
            v3b.SMALL_LESION_MAXIMUM_FRACTION
        ),
        "image_size": EXPECTED_IMAGE_SIZE,
        "prediction_threshold": EXPECTED_THRESHOLD,
        "validation_split_used": True,
        "test_split_used": False,
    }

    for name, expected in required.items():
        actual = configuration.get(name)
        matches = (
            almost_equal(actual, expected)
            if isinstance(expected, float) and actual is not None
            else actual == expected
        )
        if not matches:
            raise ValueError(
                f"Resume setting {name} is {actual!r}; expected "
                f"{expected!r}."
            )

    required_state_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "completed_epoch",
        "best_validation_selection_score",
        "best_validation_loss",
        "epochs_without_improvement",
    }
    missing = required_state_keys - set(checkpoint)
    if missing:
        raise KeyError(
            f"V3C resume checkpoint is missing: {sorted(missing)}"
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


def configure_experiment():
    """Apply V3B settings plus the single BatchNorm change."""

    v3b.SMOKE_TEST = SMOKE_TEST
    v3b.configure_experiment()

    checkpoint_directory = DRIVE_PROJECT_DIRECTORY / "checkpoints"
    training.SOURCE_CHECKPOINT_PATH = SOURCE_CHECKPOINT_PATH
    training.LAST_CHECKPOINT_PATH = checkpoint_directory / (
        "pneumothorax_512_v3c_batchnorm_stabilized_smoke_last.pth"
        if SMOKE_TEST
        else "pneumothorax_512_v3c_batchnorm_stabilized_last.pth"
    )
    training.BEST_CHECKPOINT_PATH = checkpoint_directory / (
        "pneumothorax_512_v3c_batchnorm_stabilized_smoke_best.pth"
        if SMOKE_TEST
        else "pneumothorax_512_v3c_batchnorm_stabilized_best.pth"
    )
    training.TRAINING_STAGE = (
        "pneumothorax_512_v3c_batchnorm_stabilized_smoke_test"
        if SMOKE_TEST
        else "pneumothorax_512_v3c_batchnorm_stabilized_finetune"
    )

    training.run_epoch = run_epoch_with_stable_batchnorm
    training.save_checkpoint = save_v3c_checkpoint
    training.load_resume_state = load_v3c_resume_state


def main():
    v3b.validate_inherited_v1_configuration()
    v3b.validate_source_checkpoint()
    configure_experiment()

    print("Pneumothorax Model V3C - BatchNorm stabilization experiment")
    print("------------------------------------------------------------")
    print(f"Run mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL TRAINING'}")
    print(f"Locked V1 source: {SOURCE_CHECKPOINT_PATH}")
    print("V3B sampler: unchanged")
    print("V1 loss, augmentation and optimizer: unchanged")
    print("Controlled change relative to V3B:")
    print("  BatchNorm uses saved running statistics during training")
    print("  BN affine parameters are not additionally frozen")
    print("  Running buffers are verified unchanged after each epoch")
    print("Validation split: checkpoint selection only")
    print("Test split: not created or accessed\n")

    training.main()


if __name__ == "__main__":
    main()
