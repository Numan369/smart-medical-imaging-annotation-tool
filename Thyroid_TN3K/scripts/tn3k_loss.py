import torch
import torch.nn as nn


# ============================================================
# SOFT DICE LOSS
# ============================================================

class SoftDiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Input:
        logits: [B, 1, H, W]
        targets: [B, 1, H, W], values {0,1}

    The model produces raw logits.
    Sigmoid is applied inside this loss.
    """

    def __init__(
        self,
        smooth=1e-6,
    ):

        super().__init__()

        self.smooth = smooth


    def forward(
        self,
        logits,
        targets,
    ):

        # ----------------------------------------------------
        # Convert logits -> probabilities
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            logits
        )

        targets = targets.float()


        # ----------------------------------------------------
        # Flatten each image independently
        #
        # [B,1,H,W]
        # ->
        # [B, number_of_pixels]
        # ----------------------------------------------------

        probabilities = probabilities.reshape(
            probabilities.shape[0],
            -1,
        )

        targets = targets.reshape(
            targets.shape[0],
            -1,
        )


        # ----------------------------------------------------
        # Dice components
        # ----------------------------------------------------

        intersection = (
            probabilities
            * targets
        ).sum(
            dim=1
        )


        predicted_sum = (
            probabilities.sum(
                dim=1
            )
        )


        target_sum = (
            targets.sum(
                dim=1
            )
        )


        dice = (
            (
                2.0 * intersection
                + self.smooth
            )
            /
            (
                predicted_sum
                + target_sum
                + self.smooth
            )
        )


        # ----------------------------------------------------
        # Dice loss
        #
        # Perfect overlap:
        #
        # Dice = 1
        # Loss = 0
        # ----------------------------------------------------

        dice_loss = (
            1.0
            - dice
        )


        # Average across batch.
        return dice_loss.mean()


# ============================================================
# TN3K V1 COMBINED LOSS
# ============================================================

class TN3KV1Loss(nn.Module):
    """
    TN3K V1 training objective:

        0.50 * BCEWithLogitsLoss
        +
        0.50 * Soft Dice Loss
    """

    def __init__(
        self,
        bce_weight=0.50,
        dice_weight=0.50,
    ):

        super().__init__()


        if (
            bce_weight < 0
            or dice_weight < 0
        ):

            raise ValueError(
                "Loss weights must be non-negative."
            )


        total_weight = (
            bce_weight
            + dice_weight
        )


        if total_weight <= 0:

            raise ValueError(
                "At least one loss weight "
                "must be greater than zero."
            )


        # Normalize weights.
        self.bce_weight = (
            bce_weight
            / total_weight
        )

        self.dice_weight = (
            dice_weight
            / total_weight
        )


        # IMPORTANT:
        #
        # BCEWithLogitsLoss expects RAW logits.
        # Do NOT sigmoid logits before sending them here.
        self.bce_loss = (
            nn.BCEWithLogitsLoss()
        )


        self.dice_loss = (
            SoftDiceLoss()
        )


    def forward(
        self,
        logits,
        targets,
    ):

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        if (
            logits.shape
            != targets.shape
        ):

            raise ValueError(
                "Logits and target shapes "
                "must match. "
                f"Logits={tuple(logits.shape)}, "
                f"Targets={tuple(targets.shape)}"
            )


        if not torch.isfinite(
            logits
        ).all():

            raise ValueError(
                "Logits contain NaN or Infinity."
            )


        if not torch.isfinite(
            targets
        ).all():

            raise ValueError(
                "Targets contain NaN or Infinity."
            )


        # ----------------------------------------------------
        # BCE
        # ----------------------------------------------------

        bce = self.bce_loss(
            logits,
            targets,
        )


        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        dice = self.dice_loss(
            logits,
            targets,
        )


        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        total = (
            self.bce_weight
            * bce
            +
            self.dice_weight
            * dice
        )


        # ----------------------------------------------------
        # Return both total + components.
        #
        # This is useful for training logs.
        # ----------------------------------------------------

        components = {

            "total_loss": (
                total.detach().item()
            ),

            "bce_loss": (
                bce.detach().item()
            ),

            "dice_loss": (
                dice.detach().item()
            ),
        }


        return (
            total,
            components,
        )


# ============================================================
# BASIC LOSS TEST
# ============================================================

def main():

    print("=" * 70)
    print("TN3K V1 LOSS BASIC TEST")
    print("=" * 70)


    criterion = TN3KV1Loss(
        bce_weight=0.50,
        dice_weight=0.50,
    )


    # ========================================================
    # SYNTHETIC GROUND TRUTH
    # ========================================================

    targets = torch.zeros(
        2,
        1,
        512,
        512,
        dtype=torch.float32,
    )


    # Two fake nodules.
    targets[
        0,
        0,
        180:300,
        200:330,
    ] = 1.0


    targets[
        1,
        0,
        220:310,
        150:260,
    ] = 1.0


    # ========================================================
    # TEST 1:
    # RANDOM / UNINFORMED PREDICTION
    # ========================================================

    random_logits = torch.zeros_like(
        targets
    )


    (
        random_loss,
        random_components,
    ) = criterion(
        random_logits,
        targets,
    )


    print()
    print("UNINFORMED PREDICTION")

    print(
        f"Total loss: "
        f"{random_components['total_loss']:.6f}"
    )

    print(
        f"BCE loss:   "
        f"{random_components['bce_loss']:.6f}"
    )

    print(
        f"Dice loss:  "
        f"{random_components['dice_loss']:.6f}"
    )


    # ========================================================
    # TEST 2:
    # VERY GOOD PREDICTION
    #
    # Background logits = -8
    # Nodule logits     = +8
    # ========================================================

    good_logits = torch.full_like(
        targets,
        -8.0,
    )


    good_logits = torch.where(
        targets > 0.5,
        torch.tensor(
            8.0,
            dtype=good_logits.dtype,
        ),
        good_logits,
    )


    (
        good_loss,
        good_components,
    ) = criterion(
        good_logits,
        targets,
    )


    print()
    print("GOOD PREDICTION")

    print(
        f"Total loss: "
        f"{good_components['total_loss']:.6f}"
    )

    print(
        f"BCE loss:   "
        f"{good_components['bce_loss']:.6f}"
    )

    print(
        f"Dice loss:  "
        f"{good_components['dice_loss']:.6f}"
    )


    # ========================================================
    # SAFETY TESTS
    # ========================================================

    assert torch.isfinite(
        random_loss
    )

    assert torch.isfinite(
        good_loss
    )


    # A very good segmentation MUST produce
    # substantially lower loss than an
    # uninformed segmentation.
    assert (
        good_loss
        < random_loss
    )


    # Loss needs gradients during training.
    gradient_test_logits = (
        torch.randn(
            2,
            1,
            64,
            64,
            requires_grad=True,
        )
    )


    gradient_test_targets = (
        torch.zeros_like(
            gradient_test_logits
        )
    )


    gradient_test_targets[
        :,
        :,
        20:40,
        20:40,
    ] = 1.0


    gradient_loss, _ = criterion(
        gradient_test_logits,
        gradient_test_targets,
    )


    gradient_loss.backward()


    assert (
        gradient_test_logits.grad
        is not None
    )


    assert torch.isfinite(
        gradient_test_logits.grad
    ).all()


    print()
    print(
        "Gradient backpropagation: PASSED"
    )


    print()
    print("=" * 70)
    print(
        "TN3K V1 LOSS BASIC TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()