import torch

from tn3k_dataloaders import create_tn3k_dataloaders
from tn3k_model import (
    TN3KResNet34UNet,
    count_parameters,
)
from tn3k_loss import TN3KV1Loss


# ============================================================
# TN3K TRAINING SMOKE TEST
# ============================================================
#
# PURPOSE:
#
# Verify the complete real training chain:
#
# TN3K Dataset
#     ↓
# DataLoader
#     ↓
# augmentation
#     ↓
# ResNet34 U-Net
#     ↓
# loss
#     ↓
# backward()
#     ↓
# gradients
#     ↓
# optimizer.step()
#
# This is NOT real training.
#
# Only one real training batch is used.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

# Use batch size 1 locally to reduce CPU/RAM usage.
#
# Real Colab training can use physical batch size 2.
SMOKE_BATCH_SIZE = 1

LEARNING_RATE = 1e-4


# ============================================================
# GRADIENT CHECK
# ============================================================

def inspect_gradients(model):

    parameters_with_gradient = 0
    parameters_without_gradient = 0

    total_gradient_norm = 0.0

    nonfinite_gradient_found = False

    for name, parameter in (
        model.named_parameters()
    ):

        # ----------------------------------------------------
        # Ignore intentionally frozen parameters.
        # ----------------------------------------------------

        if not parameter.requires_grad:

            continue


        # ----------------------------------------------------
        # Trainable parameter should normally get gradient.
        # ----------------------------------------------------

        if parameter.grad is None:

            parameters_without_gradient += 1

            continue


        parameters_with_gradient += 1


        # ----------------------------------------------------
        # Ensure gradient is finite.
        # ----------------------------------------------------

        if not torch.isfinite(
            parameter.grad
        ).all():

            print(
                f"NON-FINITE GRADIENT: {name}"
            )

            nonfinite_gradient_found = True


        # ----------------------------------------------------
        # Accumulate simple gradient magnitude.
        # ----------------------------------------------------

        gradient_norm = (
            parameter.grad
            .detach()
            .norm()
            .item()
        )

        total_gradient_norm += (
            gradient_norm
        )


    return {

        "with_gradient":
            parameters_with_gradient,

        "without_gradient":
            parameters_without_gradient,

        "total_gradient_norm":
            total_gradient_norm,

        "nonfinite":
            nonfinite_gradient_found,
    }


# ============================================================
# CHECK ENCODER BATCHNORM
# ============================================================

def inspect_encoder_batchnorm(model):

    batchnorm_total = 0
    batchnorm_training = 0
    batchnorm_eval = 0


    encoder_modules = [

        model.encoder_stem,
        model.encoder1,
        model.encoder2,
        model.encoder3,
        model.encoder4,

    ]


    for module in encoder_modules:

        for submodule in (
            module.modules()
        ):

            if isinstance(
                submodule,
                torch.nn.BatchNorm2d,
            ):

                batchnorm_total += 1

                if submodule.training:

                    batchnorm_training += 1

                else:

                    batchnorm_eval += 1


    return {

        "total":
            batchnorm_total,

        "training":
            batchnorm_training,

        "eval":
            batchnorm_eval,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TN3K V1 REAL TRAINING SMOKE TEST")
    print("=" * 70)


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
        f"Device: {device}"
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    print()
    print("Creating TN3K DataLoaders...")


    (
        train_loader,
        validation_loader,
    ) = create_tn3k_dataloaders(

        batch_size=SMOKE_BATCH_SIZE,

        num_workers=0,
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


    # Apply TN3K V1 freezing policy.
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
    print("MODEL PARAMETERS")
    print("-" * 70)

    print(
        f"Total:     "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Frozen:    "
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

        weight_decay=1e-4,
    )


    # ========================================================
    # GET ONE REAL TRAINING BATCH
    # ========================================================

    print()
    print(
        "Loading one REAL TN3K "
        "training batch..."
    )


    train_batch = next(
        iter(train_loader)
    )


    images = (
        train_batch["image"]
        .to(device)
    )


    masks = (
        train_batch["mask"]
        .to(device)
    )


    print()
    print("REAL TRAINING BATCH")
    print("-" * 70)

    print(
        f"Sample ID: "
        f"{list(train_batch['sample_id'])}"
    )

    print(
        f"Nodule group: "
        f"{list(train_batch['nodule_size_group'])}"
    )

    print(
        f"Image shape: "
        f"{tuple(images.shape)}"
    )

    print(
        f"Mask shape:  "
        f"{tuple(masks.shape)}"
    )

    print(
        f"Image range: "
        f"{images.min().item():.4f}"
        f" -> "
        f"{images.max().item():.4f}"
    )

    print(
        f"Mask values: "
        f"{torch.unique(masks).tolist()}"
    )

    print(
        f"Ground-truth mask pixels: "
        f"{int(masks.sum().item())}"
    )


    # ========================================================
    # TRAIN MODE
    # ========================================================

    model.train()


    # IMPORTANT:
    #
    # model.train() puts BatchNorm layers into training mode.
    #
    # Because our physical batch is very small,
    # immediately restore encoder BatchNorm to eval mode
    # so its running statistics stay stable.
    model.freeze_encoder_batchnorm_stats()


    # ========================================================
    # VERIFY BATCHNORM STATE
    # ========================================================

    bn_info = inspect_encoder_batchnorm(
        model
    )


    print()
    print("ENCODER BATCHNORM CHECK")
    print("-" * 70)

    print(
        f"Total encoder BatchNorm layers: "
        f"{bn_info['total']}"
    )

    print(
        f"In training mode: "
        f"{bn_info['training']}"
    )

    print(
        f"In eval mode:     "
        f"{bn_info['eval']}"
    )


    assert (
        bn_info["training"]
        == 0
    )

    assert (
        bn_info["eval"]
        == bn_info["total"]
    )


    # ========================================================
    # STORE ONE PARAMETER BEFORE UPDATE
    # ========================================================

    weight_before = (

        model.output_layer.weight
        .detach()
        .clone()
    )


    # ========================================================
    # ZERO OLD GRADIENTS
    # ========================================================

    optimizer.zero_grad(
        set_to_none=True
    )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    print()
    print(
        "Running forward pass..."
    )


    logits = model(
        images
    )


    print(
        f"Logits shape: "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Logits range: "
        f"{logits.min().item():.4f}"
        f" -> "
        f"{logits.max().item():.4f}"
    )


    probabilities = torch.sigmoid(
        logits
    )


    print(
        f"Probability range: "
        f"{probabilities.min().item():.4f}"
        f" -> "
        f"{probabilities.max().item():.4f}"
    )


    # ========================================================
    # LOSS
    # ========================================================

    print()
    print(
        "Calculating loss..."
    )


    (
        loss,
        components,
    ) = criterion(
        logits,
        masks,
    )


    print()
    print("LOSS")
    print("-" * 70)

    print(
        f"Total loss: "
        f"{components['total_loss']:.6f}"
    )

    print(
        f"BCE loss:   "
        f"{components['bce_loss']:.6f}"
    )

    print(
        f"Dice loss:  "
        f"{components['dice_loss']:.6f}"
    )


    # ========================================================
    # LOSS SAFETY CHECK
    # ========================================================

    assert torch.isfinite(
        loss
    )


    # ========================================================
    # BACKWARD PASS
    # ========================================================

    print()
    print(
        "Running backward pass..."
    )


    loss.backward()


    # ========================================================
    # GRADIENT INSPECTION
    # ========================================================

    gradient_info = (
        inspect_gradients(
            model
        )
    )


    print()
    print("GRADIENT CHECK")
    print("-" * 70)

    print(
        f"Trainable parameter tensors "
        f"with gradients: "
        f"{gradient_info['with_gradient']}"
    )

    print(
        f"Trainable parameter tensors "
        f"without gradients: "
        f"{gradient_info['without_gradient']}"
    )

    print(
        f"Total gradient norm: "
        f"{gradient_info['total_gradient_norm']:.6f}"
    )

    print(
        f"Non-finite gradients found: "
        f"{gradient_info['nonfinite']}"
    )


    assert not (
        gradient_info["nonfinite"]
    )


    assert (
        gradient_info[
            "with_gradient"
        ]
        > 0
    )


    assert (
        gradient_info[
            "total_gradient_norm"
        ]
        > 0.0
    )


    # ========================================================
    # OPTIMIZER STEP
    # ========================================================

    print()
    print(
        "Running optimizer step..."
    )


    optimizer.step()


    # ========================================================
    # CHECK THAT MODEL ACTUALLY CHANGED
    # ========================================================

    weight_after = (

        model.output_layer.weight
        .detach()
        .clone()
    )


    maximum_weight_change = (

        (
            weight_after
            - weight_before
        )
        .abs()
        .max()
        .item()
    )


    print()
    print("WEIGHT UPDATE CHECK")
    print("-" * 70)

    print(
        f"Maximum output-layer "
        f"weight change: "
        f"{maximum_weight_change:.10f}"
    )


    assert (
        maximum_weight_change
        > 0.0
    )


    # ========================================================
    # VERIFY FROZEN EARLY ENCODER
    # ========================================================

    frozen_gradient_error = False


    for module_name, module in [

        (
            "encoder_stem",
            model.encoder_stem,
        ),

        (
            "encoder1",
            model.encoder1,
        ),

        (
            "encoder2",
            model.encoder2,
        ),

    ]:

        for parameter in (
            module.parameters()
        ):

            if parameter.requires_grad:

                print(
                    f"ERROR: "
                    f"{module_name} contains "
                    f"a trainable parameter."
                )

                frozen_gradient_error = True


    assert not frozen_gradient_error


    print()
    print(
        "Frozen early encoder check: PASSED"
    )


    # ========================================================
    # VALIDATION FORWARD PASS
    # ========================================================

    print()
    print(
        "Testing one validation batch..."
    )


    validation_batch = next(
        iter(validation_loader)
    )


    validation_images = (

        validation_batch["image"]
        .to(device)
    )


    validation_masks = (

        validation_batch["mask"]
        .to(device)
    )


    model.eval()


    with torch.no_grad():

        validation_logits = model(
            validation_images
        )


        (
            validation_loss,
            validation_components,
        ) = criterion(
            validation_logits,
            validation_masks,
        )


    print()
    print("VALIDATION SMOKE CHECK")
    print("-" * 70)

    print(
        f"Validation image shape: "
        f"{tuple(validation_images.shape)}"
    )

    print(
        f"Validation logits shape: "
        f"{tuple(validation_logits.shape)}"
    )

    print(
        f"Validation loss: "
        f"{validation_components['total_loss']:.6f}"
    )


    assert torch.isfinite(
        validation_loss
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TN3K V1 REAL TRAINING SMOKE TEST PASSED"
    )
    print("=" * 70)

    print()
    print(
        "The complete TN3K training chain is operational:"
    )

    print()

    print(
        "Dataset -> DataLoader -> Model -> "
        "Loss -> Backward -> Optimizer"
    )


if __name__ == "__main__":
    main()