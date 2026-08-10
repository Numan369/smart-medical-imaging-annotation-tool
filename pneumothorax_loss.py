import torch
import torch.nn as nn
import torch.nn.functional as functional

from pneumothorax_dataloaders import create_dataloaders
from pneumothorax_model import PneumothoraxResNet34UNet


POSITIVE_PIXEL_WEIGHT = 10.0


class DiceLoss(nn.Module):
    """Measure foreground-mask disagreement across a batch."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(
                "Prediction and target-mask shapes must match."
            )

        probabilities = torch.sigmoid(logits)
        targets = targets.to(dtype=probabilities.dtype)

        # Calculate one foreground Dice score for the complete
        # batch. This prevents every empty image from receiving
        # its own easy perfect score.
        intersection = (probabilities * targets).sum()

        denominator = (
            probabilities.sum()
            + targets.sum()
        )

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """Combine weighted pixel loss and foreground Dice loss."""

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
        positive_pixel_weight=POSITIVE_PIXEL_WEIGHT,
    ):
        super().__init__()

        if bce_weight < 0 or dice_weight < 0:
            raise ValueError(
                "Loss weights cannot be negative."
            )

        if bce_weight + dice_weight == 0:
            raise ValueError(
                "At least one loss weight must be positive."
            )

        if positive_pixel_weight <= 0:
            raise ValueError(
                "The positive-pixel weight must be positive."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.positive_pixel_weight = float(
            positive_pixel_weight
        )

        self.register_buffer(
            "_positive_pixel_weight",
            torch.tensor(
                [self.positive_pixel_weight],
                dtype=torch.float32,
            ),
        )

        self.dice_loss = DiceLoss()

    def components(self, logits, targets):
        targets = targets.to(dtype=logits.dtype)

        positive_pixel_weight = (
            self._positive_pixel_weight.to(
                device=logits.device,
                dtype=logits.dtype,
            )
        )

        bce = functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=positive_pixel_weight,
        )

        dice = self.dice_loss(
            logits,
            targets,
        )

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
        )

        return total, bce, dice

    def forward(self, logits, targets):
        total, _, _ = self.components(
            logits,
            targets,
        )

        return total


def main():
    torch.manual_seed(42)

    data_loaders = create_dataloaders()

    training_batch = next(
        iter(data_loaders["train"])
    )

    images = training_batch["image"]
    target_masks = training_batch["mask"]
    labels = training_batch["label"]

    print("Loading pretrained segmentation model...")

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=True,
        freeze_encoder=True,
    )

    # Train the decoder while keeping the frozen encoder,
    # including its BatchNorm layers, in evaluation mode.
    model.train()

    for encoder_module in model.encoder_modules():
        encoder_module.eval()

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        positive_pixel_weight=POSITIVE_PIXEL_WEIGHT,
    )

    model.zero_grad(set_to_none=True)

    output_logits = model(images)

    total_loss, bce_loss, dice_loss = (
        criterion.components(
            output_logits,
            target_masks,
        )
    )

    if not torch.isfinite(total_loss):
        raise ValueError(
            "The calculated loss is not finite."
        )

    # Calculate gradients without updating any parameters.
    total_loss.backward()

    encoder_received_gradients = any(
        parameter.grad is not None
        for parameter in model.encoder_parameters()
    )

    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if (
            parameter.requires_grad
            and parameter.grad is not None
        )
    ]

    trainable_gradients_are_finite = (
        len(trainable_gradients) > 0
        and all(
            torch.isfinite(gradient).all().item()
            for gradient in trainable_gradients
        )
    )

    trainable_gradients_are_nonzero = any(
        torch.count_nonzero(gradient).item() > 0
        for gradient in trainable_gradients
    )

    print("\nWeighted combined-loss check")
    print("----------------------------")
    print(
        "Positive images in batch: "
        f"{int(labels.sum().item())}"
    )
    print(
        "Positive mask pixels in batch: "
        f"{int(target_masks.sum().item())}"
    )
    print(
        "Positive-pixel BCE weight: "
        f"{criterion.positive_pixel_weight:.1f}"
    )
    print(f"Output shape: {tuple(output_logits.shape)}")
    print(f"Target shape: {tuple(target_masks.shape)}")
    print(f"Weighted BCE loss: {bce_loss.item():.6f}")
    print(f"Batch Dice loss: {dice_loss.item():.6f}")
    print(f"Combined loss: {total_loss.item():.6f}")
    print(
        "Encoder received gradients: "
        f"{encoder_received_gradients}"
    )
    print(
        "Trainable gradients are finite: "
        f"{trainable_gradients_are_finite}"
    )
    print(
        "Trainable gradients are nonzero: "
        f"{trainable_gradients_are_nonzero}"
    )

    if encoder_received_gradients:
        raise ValueError(
            "The frozen encoder unexpectedly received gradients."
        )

    if not trainable_gradients_are_finite:
        raise ValueError(
            "The trainable gradients are missing or non-finite."
        )

    if not trainable_gradients_are_nonzero:
        raise ValueError(
            "The trainable gradients are all zero."
        )

    print(
        "\nWeighted loss and gradient calculation passed. "
        "No parameters were updated."
    )


if __name__ == "__main__":
    main()