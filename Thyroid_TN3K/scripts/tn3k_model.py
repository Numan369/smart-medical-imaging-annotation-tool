import torch
import torch.nn as nn

from torchvision.models import (
    resnet34,
    ResNet34_Weights,
)


# ============================================================
# DECODER BLOCK
# ============================================================

class DecoderBlock(nn.Module):
    """
    U-Net decoder block.

    Steps:

        low-resolution feature
                ↓
        upsample ×2
                ↓
        concatenate encoder skip connection
                ↓
        convolution
                ↓
        GroupNorm
                ↓
        ReLU
                ↓
        convolution
                ↓
        GroupNorm
                ↓
        ReLU

    GroupNorm is used instead of BatchNorm because
    TN3K V1 uses a physical batch size of 2.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        # ----------------------------------------------------
        # After concatenation:
        #
        # upsampled feature channels
        # +
        # skip feature channels
        # ----------------------------------------------------

        combined_channels = (
            out_channels
            + skip_channels
        )

        self.conv1 = nn.Conv2d(
            combined_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.norm1 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels,
        )

        self.relu1 = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels,
        )

        self.relu2 = nn.ReLU(
            inplace=True
        )


    def forward(
        self,
        x,
        skip,
    ):

        # ----------------------------------------------------
        # Upsample
        # ----------------------------------------------------

        x = self.upsample(x)

        # ----------------------------------------------------
        # Safety check for shape mismatch
        # ----------------------------------------------------

        if (
            x.shape[-2:]
            != skip.shape[-2:]
        ):

            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # ----------------------------------------------------
        # U-Net skip connection
        # ----------------------------------------------------

        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Convolution block
        # ----------------------------------------------------

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        return x


# ============================================================
# TN3K RESNET34 U-NET
# ============================================================

class TN3KResNet34UNet(nn.Module):

    def __init__(
        self,
        use_pretrained_encoder=True,
    ):

        super().__init__()


        # ====================================================
        # RESNET34 ENCODER
        # ====================================================

        if use_pretrained_encoder:

            weights = (
                ResNet34_Weights.DEFAULT
            )

        else:

            weights = None


        backbone = resnet34(
            weights=weights
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # TN3K image arrives as:
        #
        # [B,1,512,512]
        #
        # ResNet expects:
        #
        # [B,3,512,512]
        #
        # We will repeat grayscale channel inside forward().
        # ----------------------------------------------------


        # ====================================================
        # ENCODER STAGES
        # ====================================================

        self.encoder_stem = nn.Sequential(

            backbone.conv1,

            backbone.bn1,

            backbone.relu,
        )

        self.encoder_pool = (
            backbone.maxpool
        )


        # ResNet34 feature stages

        self.encoder1 = (
            backbone.layer1
        )

        self.encoder2 = (
            backbone.layer2
        )

        self.encoder3 = (
            backbone.layer3
        )

        self.encoder4 = (
            backbone.layer4
        )


        # ====================================================
        # DECODER
        # ====================================================

        # ResNet feature channel counts:
        #
        # encoder1 = 64
        # encoder2 = 128
        # encoder3 = 256
        # encoder4 = 512


        self.decoder4 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )


        self.decoder3 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )


        self.decoder2 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )


        # Skip from initial stem.
        self.decoder1 = DecoderBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=64,
        )


        # ====================================================
        # FINAL UPSAMPLING
        #
        # stem features are at 1/2 resolution,
        # so after decoder1 we need one final ×2 step.
        # ====================================================

        self.final_up = (
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2,
            )
        )


        self.final_conv_block = (
            nn.Sequential(

                nn.Conv2d(
                    32,
                    32,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),

                nn.GroupNorm(
                    8,
                    32,
                ),

                nn.ReLU(
                    inplace=True
                ),

                nn.Conv2d(
                    32,
                    32,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),

                nn.GroupNorm(
                    8,
                    32,
                ),

                nn.ReLU(
                    inplace=True
                ),
            )
        )


        # ====================================================
        # OUTPUT
        #
        # ONE channel because this is binary segmentation:
        #
        # background
        # vs
        # thyroid nodule
        # ====================================================

        self.output_layer = (
            nn.Conv2d(
                32,
                1,
                kernel_size=1,
            )
        )


        # ====================================================
        # IMAGENET NORMALIZATION
        # ====================================================

        self.register_buffer(

            "imagenet_mean",

            torch.tensor(
                [
                    0.485,
                    0.456,
                    0.406,
                ]
            ).view(
                1,
                3,
                1,
                1,
            ),
        )


        self.register_buffer(

            "imagenet_std",

            torch.tensor(
                [
                    0.229,
                    0.224,
                    0.225,
                ]
            ).view(
                1,
                3,
                1,
                1,
            ),
        )


    # ========================================================
    # FREEZING / FINE-TUNING POLICY
    # ========================================================

    def configure_v1_trainable_layers(
        self,
    ):
        """
        TN3K V1 fine-tuning policy.

        Frozen:
            stem
            encoder1
            encoder2

        Trainable:
            encoder3
            encoder4
            complete decoder
            output layer

        This protects early generic pretrained features
        while allowing deeper layers to adapt to thyroid
        ultrasound.
        """

        # ----------------------------------------------------
        # Freeze entire encoder first
        # ----------------------------------------------------

        for module in [

            self.encoder_stem,
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,

        ]:

            for parameter in (
                module.parameters()
            ):

                parameter.requires_grad = (
                    False
                )


        # ----------------------------------------------------
        # Unfreeze deeper blocks
        # ----------------------------------------------------

        for module in [

            self.encoder3,
            self.encoder4,

        ]:

            for parameter in (
                module.parameters()
            ):

                parameter.requires_grad = (
                    True
                )


        # Decoder is already trainable by default.


    # ========================================================
    # STABILIZE ENCODER BATCHNORM
    # ========================================================

    def freeze_encoder_batchnorm_stats(
        self,
    ):
        """
        Keep pretrained ResNet BatchNorm running statistics
        fixed.

        Important for physical batch size 2.

        This does NOT remove BatchNorm from the network.
        It simply prevents noisy small-batch running-stat
        updates.
        """

        encoder_modules = [

            self.encoder_stem,
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,

        ]


        for module in encoder_modules:

            for submodule in (
                module.modules()
            ):

                if isinstance(
                    submodule,
                    nn.BatchNorm2d,
                ):

                    submodule.eval()


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
    ):

        # ----------------------------------------------------
        # Input safety
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                "Expected input shape "
                "[B,C,H,W]. "
                f"Received {tuple(x.shape)}"
            )


        if x.shape[1] != 1:

            raise ValueError(
                "TN3K model expects "
                "1-channel grayscale input. "
                f"Received {x.shape[1]} channels."
            )


        # ====================================================
        # GRAYSCALE → 3 CHANNELS
        # ====================================================

        x = x.repeat(
            1,
            3,
            1,
            1,
        )


        # ====================================================
        # IMAGENET NORMALIZATION
        # ====================================================

        x = (
            x
            - self.imagenet_mean
        ) / self.imagenet_std


        # ====================================================
        # ENCODER
        # ====================================================

        # 512 -> 256
        stem = self.encoder_stem(x)

        # 256 -> 128
        pooled = self.encoder_pool(
            stem
        )

        # 128
        e1 = self.encoder1(
            pooled
        )

        # 128 -> 64
        e2 = self.encoder2(
            e1
        )

        # 64 -> 32
        e3 = self.encoder3(
            e2
        )

        # 32 -> 16
        e4 = self.encoder4(
            e3
        )


        # ====================================================
        # DECODER
        # ====================================================

        # 16 -> 32
        d4 = self.decoder4(
            e4,
            e3,
        )

        # 32 -> 64
        d3 = self.decoder3(
            d4,
            e2,
        )

        # 64 -> 128
        d2 = self.decoder2(
            d3,
            e1,
        )

        # 128 -> 256
        d1 = self.decoder1(
            d2,
            stem,
        )

        # 256 -> 512
        x = self.final_up(
            d1
        )

        x = self.final_conv_block(
            x
        )


        # ====================================================
        # OUTPUT LOGITS
        # ====================================================

        logits = self.output_layer(
            x
        )


        # IMPORTANT:
        #
        # NO sigmoid here.
        #
        # Training loss will use logits directly.
        #
        # During evaluation:
        #
        # probabilities = torch.sigmoid(logits)
        #

        return logits


# ============================================================
# PARAMETER COUNTS
# ============================================================

def count_parameters(
    model,
):

    total = sum(

        parameter.numel()

        for parameter
        in model.parameters()
    )


    trainable = sum(

        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )


    frozen = (
        total
        - trainable
    )


    return (
        total,
        trainable,
        frozen,
    )


# ============================================================
# BASIC MODEL TEST
# ============================================================

def main():

    print("=" * 70)
    print("TN3K RESNET34 U-NET BASIC TEST")
    print("=" * 70)


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = TN3KResNet34UNet(
        use_pretrained_encoder=True,
    )


    model.configure_v1_trainable_layers()


    # ========================================================
    # PARAMETER COUNTS
    # ========================================================

    (
        total,
        trainable,
        frozen,
    ) = count_parameters(
        model
    )


    print()
    print(
        f"Total parameters: "
        f"{total:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Frozen parameters: "
        f"{frozen:,}"
    )


    # ========================================================
    # TEST INPUT
    # ========================================================

    x = torch.rand(
        2,
        1,
        512,
        512,
    )


    print()
    print(
        f"Input shape: "
        f"{tuple(x.shape)}"
    )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    model.eval()


    with torch.no_grad():

        logits = model(x)


    print(
        f"Output logits shape: "
        f"{tuple(logits.shape)}"
    )


    probabilities = torch.sigmoid(
        logits
    )


    print(
        f"Probability minimum: "
        f"{probabilities.min().item():.4f}"
    )

    print(
        f"Probability maximum: "
        f"{probabilities.max().item():.4f}"
    )


    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    assert tuple(
        logits.shape
    ) == (
        2,
        1,
        512,
        512,
    )


    assert torch.isfinite(
        logits
    ).all()


    assert torch.isfinite(
        probabilities
    ).all()


    assert (
        probabilities.min()
        >= 0.0
    )


    assert (
        probabilities.max()
        <= 1.0
    )


    print()
    print("=" * 70)
    print(
        "TN3K MODEL BASIC TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()